"""
Diagnostics suite for prototype model.

Loads Stage B checkpoint, cache, and prototypes; then runs:
 1) Attention spread / entropy stats on prototype cross-attn.
 2) Per-head prototype attention sharpness.
 3) Occlusion test (top-K vs random prototypes).
 4) Randomized prototype tests (block shuffle + random prototypes).

Outputs are printed; metrics saved optionally via --save_json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from peer.data import build_raw_dataloaders, build_test_loader
from peer.llama_backbone import FrozenLlama
from peer.modules import (
    InferenceHead,
    PerceiverCompressor,
    ScalarLabelEmbedder,
    SlotSelector,
)
from peer.prompting import build_prompts
from peer.utils import regression_metrics


def load_stage_b_modules(ckpt, device):
    d_model = ckpt["d_model"]
    d_h = ckpt.get("d_h", 256)
    mq = ckpt.get("mq", 8)
    n_heads = ckpt.get("n_heads", 8)
    L = ckpt.get("L", 3)
    Wq = torch.nn.Linear(d_model, d_h).to(device)
    QueryComp = PerceiverCompressor(d_h=d_h, m=mq, n_heads=n_heads).to(device)
    LabelEmb = ScalarLabelEmbedder(d_h=d_h).to(device)
    Inference = InferenceHead(d_h=d_h, n_heads=n_heads, L=L).to(device)

    Wq.load_state_dict(ckpt["Wq"])
    QueryComp.load_state_dict(ckpt["QueryComp"])
    LabelEmb.load_state_dict(ckpt["LabelEmb"])
    Inference.load_state_dict(ckpt["InferenceHead"])
    LabelEmb.set_stats(ckpt.get("label_mean", 0.0), ckpt.get("label_std", 1.0))
    return Wq, QueryComp, LabelEmb, Inference, d_h


def build_fixed_memory(mem_cache, y_cache, prototypes, label_emb):
    """
    Build fixed prototype memory tensor [1, K*(m+1), d_h]
    """
    idx = prototypes
    Mem_sel = mem_cache[idx].float()  # [K,m,d_h]
    Ltok = label_emb(y_cache[idx]).squeeze(1)  # [K,d_h]
    m = Mem_sel.size(1)
    Mem_slot = torch.cat([Mem_sel, Ltok.unsqueeze(1)], dim=1)  # [K,m+1,d_h]
    Mem_flat_fixed = Mem_slot.view(1, idx.numel() * (m + 1), Mem_sel.size(2))
    return Mem_flat_fixed, m, idx


def run_with_attn(inference: InferenceHead, Qq, Mem_flat):
    """
    Custom forward to grab per-head attention probs from the last layer.
    """
    R = inference.R0.unsqueeze(0).expand(Qq.size(0), 1, -1)
    attn_last = None
    for layer in inference.layers:
        R2, _ = layer.attn_q(R, Qq, Qq, need_weights=False)
        R = layer.ln1(R + R2)
        R2, attn = layer.attn_m(
            R, Mem_flat, Mem_flat, need_weights=True, average_attn_weights=False
        )
        R = layer.ln2(R + R2)
        R = layer.ln3(R + layer.ff(R))
        attn_last = attn  # [B, H, tgt_len, src_len]
    raw = inference.out(R[:, 0, :])  # [B,1]
    return raw, attn_last  # raw logits, per-head attn


def attention_block_stats(attn_probs: torch.Tensor, K: int, block: int) -> Dict:
    """
    attn_probs: [B, H, 1, L], L=K*block
    Returns per-head aggregated prototype attention [B,H,K] and summary stats.
    """
    B, H, _, L = attn_probs.shape
    attn_flat = attn_probs.squeeze(2)  # [B,H,L]
    # aggregate token -> prototype
    attn_proto = attn_flat.view(B, H, K, block).sum(dim=-1)  # [B,H,K]
    # stats
    entropy = -(attn_flat * (attn_flat + 1e-9).log()).sum(dim=-1)  # [B,H]
    std_logits = attn_flat.std(dim=-1)  # proxy since logits unavailable
    return {
        "attn_proto": attn_proto,
        "entropy": entropy,
        "std_logits_proxy": std_logits,
    }


def logit_spread_report(stats: Dict):
    entropy = stats["entropy"]  # [B,H]
    std_logits = stats["std_logits_proxy"]  # [B,H]
    ent_mean = entropy.mean().item()
    std_mean = std_logits.mean().item()
    std_std = std_logits.std().item()
    print(
        f"[Logit spread] mean entropy={ent_mean:.4f} (uniform~logL), std_logits_proxy mean={std_mean:.4f} std={std_std:.4f}"
    )


def per_head_report(attn_proto: torch.Tensor):
    """
    attn_proto: [B,H,K]
    """
    B, H, K = attn_proto.shape
    for h in range(H):
        A = attn_proto[:, h, :]
        top1 = A.max(dim=-1).values
        top5 = torch.topk(A, k=min(5, K), dim=-1).values.sum(dim=-1)
        ent = -(A * (A + 1e-9).log()).sum(dim=-1)
        print(
            f"[Head {h}] top1={top1.mean():.4f} top5_sum={top5.mean():.4f} entropy={ent.mean():.4f} (uniform~{torch.log(torch.tensor(float(K))).item():.4f})"
        )


def occlusion_test(Qq, Mem_flat, inference, top_proto_idx, block):
    """
    Occlude prototypes by zeroing blocks. top_proto_idx: list[int] to test.
    """
    B = Qq.size(0)
    base_raw, _ = run_with_attn(inference, Qq, Mem_flat)
    base = base_raw.squeeze(-1)

    deltas = []
    for k in top_proto_idx:
        mem_masked = Mem_flat.clone()
        start, end = k * block, (k + 1) * block
        mem_masked[:, start:end, :] = 0
        raw_masked, _ = run_with_attn(inference, Qq, mem_masked)
        delta = (base - raw_masked.squeeze(-1)).abs()
        deltas.append(delta)
    if deltas:
        deltas = torch.stack(deltas, dim=0)  # [n_proto,B]
        return deltas.mean(dim=1).cpu()
    return None


def eval_metrics(inference, Wq, QueryComp, Mem_fixed, llama, loader, device):
    preds_all, labels_all, raw_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            prompts = build_prompts(batch["text"], dataset_name=args.dataset)
            labels = batch["labels"].to(device).float()
            Hq = llama.encode(
                prompts, max_length=args.max_length, already_prompted=True
            ).float()
            Qq = QueryComp(Wq(Hq))
            Mem_flat = Mem_fixed.expand(Qq.size(0), -1, -1)
            y_hat, attn = inference(Qq, Mem_flat)
            preds_all.append(y_hat.squeeze(-1).cpu())
            labels_all.append(labels.cpu())
            raw_all.append(y_hat.squeeze(-1).cpu())
    preds_cat = torch.cat(preds_all)
    labels_cat = torch.cat(labels_all)
    raw_cat = torch.cat(raw_all)
    metrics = regression_metrics(preds_cat, labels_cat)
    metrics.update(
        {
            "yhat_mean": preds_cat.mean().item(),
            "yhat_std": preds_cat.std().item(),
            "yhat_min": preds_cat.min().item(),
            "yhat_max": preds_cat.max().item(),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Prototype diagnostics")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Samples for diagnostics batches.",
    )
    parser.add_argument(
        "--save_json",
        default=None,
        help="Optional path to save diagnostic results.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    Wq, QueryComp, LabelEmb, Inference, d_h = load_stage_b_modules(ckpt, device)
    Wq.eval()
    QueryComp.eval()
    LabelEmb.eval()
    Inference.eval()

    mem_cache = torch.load(
        os.path.join(args.cache_dir, "cache_mem.pt"), map_location="cpu"
    ).to(device)
    y_cache = torch.load(
        os.path.join(args.cache_dir, "cache_y.pt"), map_location="cpu"
    ).to(device)
    prototypes = torch.load(args.prototypes, map_location="cpu").long()

    Mem_fixed, m, proto_idx = build_fixed_memory(
        mem_cache, y_cache, prototypes, LabelEmb
    )
    block = m + 1
    K = prototypes.numel()

    llama = FrozenLlama(args.model_name, device)
    _, _, dm = build_raw_dataloaders(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        combine_fields=False,
    )
    test_loader = build_test_loader(dm, args.batch_size)

    # Quick sanity checks on training data
    train_ds = dm.dataset["train"]
    sample_idxs = random.sample(range(len(train_ds)), k=min(5, len(train_ds)))
    print(
        "\n[Sanity] 5 random training samples (text + token lengths after prompting):"
    )
    for idx in sample_idxs:
        row = train_ds[idx]
        text = row["text"]
        prompt = build_prompts([text], dataset_name=args.dataset)[0]
        tokens = llama.tok(prompt, truncation=True, max_length=args.max_length)
        print(f"idx={idx} len={len(tokens['input_ids'])} text={text}")
    labels_train = torch.tensor(train_ds["labels"], dtype=torch.float32)
    y_min, y_mean, y_max, y_std = (
        labels_train.min().item(),
        labels_train.mean().item(),
        labels_train.max().item(),
        labels_train.std().item(),
    )
    print(
        f"[Sanity] y_train min/mean/max/std = {y_min:.4f} / {y_mean:.4f} / {y_max:.4f} / {y_std:.4f}"
    )
    if dm.eval_splits:
        val_split = dm.eval_splits[0]
        labels_val = torch.tensor(
            dm.dataset[val_split]["labels"], dtype=torch.float32
        )
        print(
            f"[Sanity] y_val   min/mean/max/std = {labels_val.min().item():.4f} / {labels_val.mean().item():.4f} / {labels_val.max().item():.4f} / {labels_val.std().item():.4f}"
        )

    # Take a small batch for attention diagnostics
    batch = next(iter(test_loader))
    prompts = build_prompts(batch["text"], dataset_name=args.dataset)
    labels = batch["labels"].to(device).float()
    Hq = llama.encode(
        prompts, max_length=args.max_length, already_prompted=True
    ).float()
    Qq = QueryComp(Wq(Hq))
    Mem_flat = Mem_fixed.expand(Qq.size(0), -1, -1)

    with torch.no_grad():
        raw_out, attn = run_with_attn(Inference, Qq, Mem_flat)
    attn_probs = attn  # [B,H,1,L]
    stats = attention_block_stats(attn_probs, K=K, block=block)
    logit_spread_report(stats)
    per_head_report(stats["attn_proto"])

    # Occlusion test
    attn_proto_mean = stats["attn_proto"].mean(dim=1)  # [B,K] averaged heads
    topk = (
        torch.topk(attn_proto_mean, k=min(10, K), dim=-1)
        .indices.unique()
        .tolist()
    )
    randk = random.sample(range(K), k=min(10, K))
    top_deltas = occlusion_test(Qq, Mem_flat, Inference, topk, block)
    rand_deltas = occlusion_test(Qq, Mem_flat, Inference, randk, block)
    if top_deltas is not None:
        print(
            f"[Occlusion] mean Δ topK={top_deltas.mean():.4f} randK={rand_deltas.mean():.4f}"
        )

    # Randomized prototypes tests
    metrics_base = eval_metrics(
        Inference, Wq, QueryComp, Mem_fixed, llama, test_loader, device
    )
    print(f"[Metrics base] {metrics_base}")

    # Shuffle blocks
    perm = torch.randperm(K)
    Mem_perm = Mem_flat.view(1, K, block, d_h)[:, perm].reshape(
        1, K * block, d_h
    )
    metrics_perm = eval_metrics(
        Inference, Wq, QueryComp, Mem_perm, llama, test_loader, device
    )
    print(f"[Metrics permuted blocks] {metrics_perm}")

    # Random prototypes
    rand_idx = torch.tensor(
        random.sample(range(mem_cache.size(0)), k=K), device=device
    )
    Mem_rand, _, _ = build_fixed_memory(mem_cache, y_cache, rand_idx, LabelEmb)
    metrics_rand = eval_metrics(
        Inference, Wq, QueryComp, Mem_rand, llama, test_loader, device
    )
    print(f"[Metrics random prototypes] {metrics_rand}")

    results = {
        "logit_spread_entropy_mean": stats["entropy"].mean().item(),
        "logit_spread_std_mean": stats["std_logits_proxy"].mean().item(),
        "occlusion_top_mean_delta": (
            top_deltas.mean().item() if top_deltas is not None else None
        ),
        "occlusion_rand_mean_delta": (
            rand_deltas.mean().item() if rand_deltas is not None else None
        ),
        "metrics_base": metrics_base,
        "metrics_perm": metrics_perm,
        "metrics_rand": metrics_rand,
    }
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved diagnostics to {args.save_json}")


if __name__ == "__main__":
    main()
