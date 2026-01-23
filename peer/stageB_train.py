import argparse
import os
import sys
import json
import random
import hashlib

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from peer.data import build_raw_dataloaders, prepare_prompts
from peer.llama_backbone import FrozenLlama
from peer.modules import (
    InferenceHead,
    PerceiverCompressor,
    ScalarLabelEmbedder,
    SlotSelector,
)
from peer.utils import huber_loss, regression_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hash_state_dict(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if torch.is_tensor(v):
            h.update(v.cpu().numpy().tobytes())
        else:
            h.update(str(v).encode("utf-8"))
    return h.hexdigest()


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def load_stage_a_for_b(ckpt, device):
    d_model = ckpt["d_model"]
    d_h = ckpt.get("d_h", 256)
    mq = ckpt.get("mq", 8)
    n_heads = ckpt.get("n_heads", 8)
    L = ckpt.get("L", 3)
    Wq = nn.Linear(d_model, d_h).to(device)
    QueryComp = PerceiverCompressor(d_h=d_h, m=mq, n_heads=n_heads).to(device)
    LabelEmb = ScalarLabelEmbedder(d_h=d_h).to(device)
    Inference = InferenceHead(d_h=d_h, n_heads=n_heads, L=L).to(device)

    Wq.load_state_dict(ckpt["Wq"])
    QueryComp.load_state_dict(ckpt["QueryComp"])
    LabelEmb.load_state_dict(ckpt["LabelEmb"])
    Inference.load_state_dict(ckpt["InferenceHead"])
    LabelEmb.set_stats(ckpt.get("label_mean", 0.0), ckpt.get("label_std", 1.0))

    return Wq, QueryComp, LabelEmb, Inference, d_model, d_h


def main():
    parser = argparse.ArgumentParser(
        description="Stage B training with slot selection."
    )
    parser.add_argument("--model_name", required=True, help="HF LLaMA name.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--ckpt", default="stageA.pt", help="Path to Stage A checkpoint."
    )
    parser.add_argument(
        "--cache_dir",
        default="cache",
        help="Directory containing cache_mem.pt/keys/y.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Stage B epochs (default 10)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate (AdamW)."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Prompt tokenization max length.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=128,
        help="Number of prototype slots.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=512,
        help="Top-T candidate pool per slot.",
    )
    parser.add_argument(
        "--tau_start",
        type=float,
        default=1.0,
        help="Starting Gumbel temperature.",
    )
    parser.add_argument(
        "--tau_final",
        type=float,
        default=0.1,
        help="Final Gumbel temperature.",
    )
    parser.add_argument(
        "--lambda_ov",
        type=float,
        default=1.0,
        help="Overlap penalty weight.",
    )
    parser.add_argument(
        "--lambda_rep",
        type=float,
        default=0.1,
        help="Repulsion penalty weight.",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=0,
        help="Number of initial epochs to freeze Wq/QueryComp/Inference (selector only).",
    )
    parser.add_argument(
        "--no_regularizer_epochs",
        type=int,
        default=0,
        help="Epochs to keep lambda_ov/lambda_rep at 0 before ramping to target.",
    )
    parser.add_argument(
        "--save_dir",
        default=".",
        help="Directory to save Stage B checkpoint.",
    )
    parser.add_argument(
        "--ckpt_name",
        default="stageB.pt",
        help="Filename for Stage B checkpoint inside save_dir.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Repulsion margin on cosine similarity.",
    )
    parser.add_argument(
        "--no_bound_train",
        action="store_true",
        help="(Deprecated: outputs are unbounded now)",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="Huber delta for regression loss.",
    )
    parser.add_argument(
        "--no_standardize_labels",
        action="store_false",
        dest="standardize_labels",
        default=True,
        help="Disable label standardization (enabled by default).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--train_label_emb",
        action="store_true",
        help="If set, train LabelEmb during Stage B (default: frozen).",
    )
    parser.add_argument(
        "--best_metric",
        choices=["val_loss", "mse", "rmse", "pearson", "spearman", "kendall"],
        default="val_loss",
        help="Metric to select best checkpoint (min for val_loss/mse/rmse, max for correlations).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    cache_mem = torch.load(
        os.path.join(args.cache_dir, "cache_mem.pt"), map_location="cpu"
    ).to(device)
    cache_keys = torch.load(
        os.path.join(args.cache_dir, "cache_keys.pt"), map_location="cpu"
    ).to(device)
    cache_y = torch.load(
        os.path.join(args.cache_dir, "cache_y.pt"), map_location="cpu"
    ).to(device)
    meta_path = os.path.join(args.cache_dir, "cache_meta.json")
    cache_meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            cache_meta = json.load(f)

    ckpt_a = torch.load(args.ckpt, map_location="cpu")
    # cache validation
    if cache_meta is not None:
        expected_ckpt = cache_meta.get("stageA_ckpt", None)
        if expected_ckpt and os.path.abspath(args.ckpt) != expected_ckpt:
            raise ValueError(
                f"Cache built from {expected_ckpt}, but Stage B is loading {os.path.abspath(args.ckpt)}."
            )
        # hash consistency
        for name, key in [
            ("Wm", "Wm_hash"),
            ("MemComp", "MemComp_hash"),
            ("KeyReadout", "KeyReadout_hash"),
        ]:
            if key in cache_meta and name in ckpt_a:
                h = hash_state_dict(ckpt_a[name])
                if h != cache_meta[key]:
                    raise ValueError(
                        f"Cache {key} mismatch for {name}; rebuild cache with matching Stage A checkpoint."
                    )
        if cache_meta.get("standardize_labels") is not None:
            if bool(cache_meta["standardize_labels"]) != bool(
                ckpt_a.get("standardize_labels", True)
            ):
                raise ValueError(
                    "Cache standardize_labels flag differs from Stage A checkpoint; rebuild cache."
                )
    Wq, QueryComp, LabelEmb, Inference, d_model, d_h = load_stage_a_for_b(
        ckpt_a, device
    )
    selector = SlotSelector(K=args.K, d_h=d_h, T=args.T).to(device)
    # Warm start slot_q from random cache keys
    with torch.no_grad():
        idx_init = torch.randperm(cache_keys.size(0), device=device)[: args.K]
        selector.slot_q.copy_(cache_keys[idx_init].float())

    if bool(ckpt_a.get("standardize_labels", True)) != bool(
        args.standardize_labels
    ):
        raise ValueError(
            "standardize_labels mismatch: Stage A checkpoint "
            f"{ckpt_a.get('standardize_labels', True)} vs Stage B arg {args.standardize_labels}"
        )

    # Explicitly mark trainable components for Stage B
    train_label_emb = bool(args.train_label_emb)
    for mod in (Wq, QueryComp, Inference, selector):
        set_requires_grad(mod, True)
    set_requires_grad(LabelEmb, train_label_emb)

    params = [
        {
            "params": selector.parameters(),
            "lr": args.lr,
            "weight_decay": 0.0,
        },
        {
            "params": list(Wq.parameters())
            + list(QueryComp.parameters())
            + list(Inference.parameters()),
            "lr": args.lr * 0.1,
            "weight_decay": args.weight_decay,
        },
    ]
    if train_label_emb:
        params[1]["params"].extend(list(LabelEmb.parameters()))

    optim = torch.optim.AdamW(params, weight_decay=0.0)

    llama = FrozenLlama(args.model_name, device)
    train_loader, val_loader, _ = build_raw_dataloaders(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        combine_fields=False,
    )
    # Label stats sanity
    train_labels = torch.tensor(
        train_loader.dataset["labels"], dtype=torch.float32
    )
    print(
        f"[StageB] label stats train mean={train_labels.mean().item():.4f} std={train_labels.std().item():.4f} "
        f"min={train_labels.min().item():.4f} max={train_labels.max().item():.4f}"
    )
    # Quick input/tokenization sanity on first batch
    first_batch = next(iter(train_loader))
    sample_texts = (
        first_batch["text"]
        if "text" in first_batch
        else first_batch.get("input_text", [])
    )
    print(
        "[StageB] SAMPLE TEXTS:\n"
        + "\n---\n".join([str(t) for t in sample_texts[:3]])
    )
    tok_dbg = llama.tok(
        sample_texts[:8],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    print(
        "[StageB] token lengths:", tok_dbg["attention_mask"].sum(dim=1).tolist()
    )
    print(
        "[StageB] unique input_ids per example:",
        [ids.unique().numel() for ids in tok_dbg["input_ids"]],
    )
    if tok_dbg["input_ids"].size(0) >= 2:
        eq01 = torch.equal(tok_dbg["input_ids"][0], tok_dbg["input_ids"][1])
        print("[StageB] example0 == example1 input_ids?", eq01)

    tau_start, tau_final = args.tau_start, args.tau_final

    best_metric_name = args.best_metric
    minimize = best_metric_name in ["val_loss", "mse", "rmse"]
    best_score = float("inf") if minimize else -float("inf")
    best_ckpt = None
    best_path = None

    for epoch in range(args.epochs):
        selector.train()
        freeze_head = epoch < args.freeze_epochs
        if freeze_head:
            set_requires_grad(Wq, False)
            set_requires_grad(QueryComp, False)
            set_requires_grad(Inference, False)
            Wq.eval(), QueryComp.eval(), Inference.eval()
        else:
            set_requires_grad(Wq, True)
            set_requires_grad(QueryComp, True)
            set_requires_grad(Inference, True)
            Wq.train(), QueryComp.train(), Inference.train()
        if train_label_emb:
            LabelEmb.train()
        else:
            LabelEmb.eval()
        tau = tau_start - (epoch / max(args.epochs - 1, 1)) * (
            tau_start - tau_final
        )

        total_loss = 0.0
        n_steps = 0
        for batch in tqdm(
            train_loader,
            desc=f"StageB Epoch {epoch+1}/{args.epochs}",
            leave=False,
        ):
            prompts = prepare_prompts(batch["text"], dataset_name=args.dataset)
            labels = batch["labels"].to(device).float()

            Hq = llama.encode(
                prompts, max_length=args.max_length, already_prompted=True
            ).float()
            Qq = QueryComp(Wq(Hq))

            Mem_sel, idx_hard, logits_full, q_probs, q_st, exp_keys = selector(
                cache_keys, cache_mem, cache_y, tau
            )

            label_emb_all = LabelEmb(cache_y).squeeze(1)  # [N, d_h]
            Ltok = torch.matmul(q_st, label_emb_all)  # [K, d_h]
            m = Mem_sel.size(1)
            Mem_slot = torch.cat(
                [Mem_sel, Ltok.unsqueeze(1)], dim=1
            )  # [K, m+1, d_h]
            Mem_flat = Mem_slot.view(1, selector.K * (m + 1), d_h).expand(
                Qq.size(0), -1, -1
            )

            if args.standardize_labels:
                y_std = LabelEmb.y_std.item() + 1e-8
                y_mean = LabelEmb.y_mean.item()
                z_target = (labels - y_mean) / y_std
                z_hat, _ = Inference(Qq, Mem_flat)
                L_reg = huber_loss(
                    z_hat.squeeze(-1), z_target, delta=args.huber_delta
                )
                preds_rescaled = z_hat.squeeze(-1) * y_std + y_mean
            else:
                z_hat, _ = Inference(Qq, Mem_flat)
                L_reg = huber_loss(
                    z_hat.squeeze(-1), labels, delta=args.huber_delta
                )
                preds_rescaled = z_hat.squeeze(-1)

            if selector.K > 1:
                pairwise = torch.matmul(q_probs, q_probs.transpose(0, 1))
                mask = torch.triu(torch.ones_like(pairwise), diagonal=1).bool()
                L_ov = (
                    pairwise[mask].mean()
                    if mask.any()
                    else torch.tensor(0.0, device=device)
                )
            else:
                L_ov = torch.tensor(0.0, device=device)

            if selector.K > 1:
                k_norm = SlotSelector.l2norm(exp_keys)
                S = k_norm @ k_norm.T
                mask = torch.triu(torch.ones_like(S), diagonal=1).bool()
                rep = torch.relu(S - args.margin)
                L_rep = (
                    rep[mask].mean()
                    if mask.any()
                    else torch.tensor(0.0, device=device)
                )
            else:
                L_rep = torch.tensor(0.0, device=device)

            lam_ov = (
                args.lambda_ov if epoch >= args.no_regularizer_epochs else 0.0
            )
            lam_rep = (
                args.lambda_rep if epoch >= args.no_regularizer_epochs else 0.0
            )

            loss = L_reg + lam_ov * L_ov + lam_rep * L_rep

            optim.zero_grad()
            loss.backward()
            if n_steps == 0:

                def gn(p):
                    return None if p.grad is None else p.grad.data.norm().item()

                gWq = gn(Wq.weight)
                gR0 = gn(Inference.R0)
                gOut = gn(Inference.out[0].weight)
                gSlot = gn(selector.slot_q)

                def fmt(val):
                    return f"{val:.4e}" if val is not None else "None"

                print(
                    "[StageB] grad norms: "
                    f"Wq={fmt(gWq)} "
                    f"r0={fmt(gR0)} "
                    f"out={fmt(gOut)} "
                    f"slot_q={fmt(gSlot)}"
                )
            optim.step()

            total_loss += loss.item()
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)

        val_loss = None
        val_metrics = None
        if val_loader is not None:
            selector.eval()
            Wq.eval(), QueryComp.eval(), LabelEmb.eval(), Inference.eval()
            with torch.no_grad():

                def run_val(current_tau):
                    v_total = 0.0
                    v_steps = 0
                    preds_all = []
                    labels_all = []
                    raw_all = []
                    for batch in tqdm(
                        val_loader, desc="StageB Val", leave=False
                    ):
                        prompts = prepare_prompts(
                            batch["text"], dataset_name=args.dataset
                        )
                        labels = batch["labels"].to(device).float()
                        Hq = llama.encode(
                            prompts,
                            max_length=args.max_length,
                            already_prompted=True,
                        ).float()
                        Qq = QueryComp(Wq(Hq))
                        Mem_sel, idx_hard, _, q_probs, q_st, exp_keys = (
                            selector(
                                cache_keys, cache_mem, cache_y, current_tau
                            )
                        )
                        label_emb_all = LabelEmb(cache_y).squeeze(1)
                        Ltok = torch.matmul(q_st, label_emb_all)
                        m = Mem_sel.size(1)
                        Mem_slot = torch.cat(
                            [Mem_sel, Ltok.unsqueeze(1)], dim=1
                        )
                        Mem_flat = Mem_slot.view(
                            1, selector.K * (m + 1), d_h
                        ).expand(Qq.size(0), -1, -1)
                        if args.standardize_labels:
                            y_std = LabelEmb.y_std.item() + 1e-8
                            y_mean = LabelEmb.y_mean.item()
                            z_target = (labels - y_mean) / y_std
                            z_hat, _ = Inference(Qq, Mem_flat)
                            pred = z_hat.squeeze(-1) * y_std + y_mean
                            v_total += huber_loss(
                                z_hat.squeeze(-1),
                                z_target,
                                delta=args.huber_delta,
                            ).item()
                            raw_all.append(z_hat.squeeze(-1).cpu())
                        else:
                            z_hat, _ = Inference(Qq, Mem_flat)
                            pred = z_hat.squeeze(-1)
                            v_total += huber_loss(
                                pred, labels, delta=args.huber_delta
                            ).item()
                            raw_all.append(pred.cpu())
                        preds_all.append(pred.cpu())
                        labels_all.append(labels.cpu())
                        v_steps += 1
                    v_loss = v_total / max(v_steps, 1)
                    if preds_all:
                        val_metrics_local = regression_metrics(
                            torch.cat(preds_all), torch.cat(labels_all)
                        )
                        raw_cat = torch.cat(raw_all)
                        yhat_cat = torch.cat(preds_all)
                        labels_cat = torch.cat(labels_all)
                        y_min, y_max = (
                            labels_cat.min().item(),
                            labels_cat.max().item(),
                        )
                        denom = max(y_max - y_min, 1e-6)
                        sat_rate = (
                            (yhat_cat < y_min + 0.01 * denom)
                            | (yhat_cat > y_max - 0.01 * denom)
                        ).float().mean().item() * 100.0
                        raw_min, raw_max = (
                            raw_cat.min().item(),
                            raw_cat.max().item(),
                        )
                        raw_std = raw_cat.std().item()
                        raw_mean = raw_cat.mean().item()
                        stats_msg = (
                            f"val(tau={current_tau:.3f}) std(yhat)={yhat_cat.std().item():.4f} "
                            f"min/max(yhat)={[yhat_cat.min().item(), yhat_cat.max().item()]} "
                            f"mean(yhat)={yhat_cat.mean().item():.4f} mean(y)={labels_cat.mean().item():.4f} "
                            f"sat%={sat_rate:.2f} "
                            f"raw std/mean/min/max={raw_std:.4f}/{raw_mean:.4f}/{raw_min:.4f}/{raw_max:.4f}"
                        )
                        print(stats_msg)
                    else:
                        val_metrics_local = None
                    return v_loss, val_metrics_local

                val_loss, val_metrics = run_val(tau)
                current_score = None
                if best_metric_name == "val_loss":
                    current_score = val_loss
                elif val_metrics is not None:
                    current_score = val_metrics.get(best_metric_name)

                better = False
                if current_score is not None and current_score == current_score:
                    if minimize:
                        better = current_score < best_score
                    else:
                        better = current_score > best_score

                if better:
                    best_score = current_score
                    best_ckpt = {
                        "model_name": args.model_name,
                        "d_model": d_model,
                        "d_h": d_h,
                        "mq": QueryComp.latents.size(0),
                        "n_heads": QueryComp.attn.num_heads,
                        "L": len(Inference.layers),
                        "standardize_labels": args.standardize_labels,
                        "Wq": Wq.state_dict(),
                        "QueryComp": QueryComp.state_dict(),
                        "LabelEmb": LabelEmb.state_dict(),
                        "InferenceHead": Inference.state_dict(),
                        "SlotSelector": selector.state_dict(),
                        "label_mean": ckpt_a.get("label_mean", 0.0),
                        "label_std": ckpt_a.get("label_std", 1.0),
                        "K": args.K,
                        "T": args.T,
                        "tau_start": args.tau_start,
                        "tau_final": args.tau_final,
                        "lambda_ov": args.lambda_ov,
                        "lambda_rep": args.lambda_rep,
                        "best_metric": best_metric_name,
                        "best_score": best_score,
                    }
                    best_path = os.path.join(args.save_dir, args.ckpt_name)
                    torch.save(best_ckpt, best_path)
                    print(
                        f"Saved new best Stage B to {os.path.abspath(best_path)} ({best_metric_name}={best_score:.4f})"
                    )

        print(
            f"Epoch {epoch+1}/{args.epochs} - tau={tau:.3f} "
            f"train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
        )
        if val_metrics is not None:
            print(
                "Val metrics: "
                + ", ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items() if v == v
                )
            )

    if best_ckpt is None:
        ckpt_b = {
            "model_name": args.model_name,
            "d_model": d_model,
            "d_h": d_h,
            "mq": QueryComp.latents.size(0),
            "n_heads": QueryComp.attn.num_heads,
            "L": len(Inference.layers),
            "standardize_labels": args.standardize_labels,
            "Wq": Wq.state_dict(),
            "QueryComp": QueryComp.state_dict(),
            "LabelEmb": LabelEmb.state_dict(),
            "InferenceHead": Inference.state_dict(),
            "SlotSelector": selector.state_dict(),
            "label_mean": ckpt_a.get("label_mean", 0.0),
            "label_std": ckpt_a.get("label_std", 1.0),
            "K": args.K,
            "T": args.T,
            "tau_start": args.tau_start,
            "tau_final": args.tau_final,
            "lambda_ov": args.lambda_ov,
            "lambda_rep": args.lambda_rep,
            "best_metric": best_metric_name,
            "best_score": (
                best_score
                if best_score
                not in (float("inf") if minimize else -float("inf"))
                else None
            ),
        }
        ckpt_out = os.path.join(args.save_dir, args.ckpt_name)
        torch.save(ckpt_b, ckpt_out)
        print(
            f"Saved Stage B checkpoint to {os.path.abspath(ckpt_out)} (no val improvement)"
        )
    else:
        print(
            f"Best Stage B checkpoint already saved at {os.path.abspath(best_path) if best_path else args.ckpt_name}"
        )


if __name__ == "__main__":
    main()
