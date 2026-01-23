import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from peer.data import (
    build_raw_dataloaders,
    build_test_loader,
    prepare_prompts,
)
from peer.llama_backbone import FrozenLlama
from peer.modules import InferenceHead, PerceiverCompressor, ScalarLabelEmbedder
from peer.utils import regression_metrics


def load_stage_b_modules(ckpt, device):
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
    return Wq, QueryComp, LabelEmb, Inference, d_h


def main():
    parser = argparse.ArgumentParser(
        description="Inference over test split with fixed prototypes."
    )
    parser.add_argument("--model_name", required=True, help="HF LLaMA name.")
    parser.add_argument(
        "--dataset", required=True, help="Dataset name for prompt selection."
    )
    parser.add_argument(
        "--cache_dir", default="cache", help="Directory with cache tensors."
    )
    parser.add_argument(
        "--ckpt", default="stageC.pt", help="Path to Stage B/C checkpoint."
    )
    parser.add_argument(
        "--prototypes",
        default="prototypes.pt",
        help="Prototype indices tensor.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Prompt tokenization max length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for test inference.",
    )
    parser.add_argument(
        "--save_dir",
        default=".",
        help="Directory to save metrics and explanations.",
    )
    parser.add_argument(
        "--metrics_name",
        default="metrics.json",
        help="Filename for metrics JSON.",
    )
    parser.add_argument(
        "--explanations_name",
        default="explanations.json",
        help="Filename for sampled explanations JSON.",
    )
    parser.add_argument(
        "--run_val",
        action="store_true",
        help="Also run evaluation on the validation split for sanity vs Stage C.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    Wq, QueryComp, LabelEmb, Inference, d_h = load_stage_b_modules(ckpt, device)
    Wq.eval(), QueryComp.eval(), LabelEmb.eval(), Inference.eval()
    torch.set_grad_enabled(False)
    standardize = bool(ckpt.get("standardize_labels", True))
    y_std = LabelEmb.y_std.item() + 1e-8
    y_mean = LabelEmb.y_mean.item()

    mem_cache = torch.load(
        os.path.join(args.cache_dir, "cache_mem.pt"), map_location="cpu"
    ).to(device)
    y_cache = torch.load(
        os.path.join(args.cache_dir, "cache_y.pt"), map_location="cpu"
    ).to(device)
    with open(os.path.join(args.cache_dir, "cache_text.json"), "r") as f:
        text_cache = json.load(f)
    prototypes = torch.load(args.prototypes, map_location="cpu").long()

    llama = FrozenLlama(args.model_name, device)

    idx = prototypes
    Mem_sel = mem_cache[idx].float()  # [K, m, d_h]
    y_proto = y_cache[idx]  # [K]
    Ltok = LabelEmb(y_proto).squeeze(1)  # [K, d_h]
    m = Mem_sel.size(1)
    Mem_slot = torch.cat([Mem_sel, Ltok.unsqueeze(1)], dim=1)  # [K, m+1, d_h]
    base_mem_flat = Mem_slot.view(1, idx.numel() * (m + 1), d_h)
    if "K" in ckpt and int(ckpt["K"]) != int(idx.numel()):
        raise ValueError(
            f"Prototype count mismatch: ckpt K={ckpt['K']} vs prototypes {idx.numel()}"
        )
    if "m" in ckpt and int(ckpt["m"]) != int(m):
        raise ValueError(
            f"Memory tokens mismatch: ckpt m={ckpt['m']} vs cache m={m}"
        )

    # Build datamodule and test loader
    train_loader, val_loader, dm = build_raw_dataloaders(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        combine_fields=False,
    )
    test_loader = build_test_loader(dm, args.batch_size)

    def run_eval(dataloader, split_name):
        preds_all = []
        labels_all = []
        for batch in dataloader:
            prompts = prepare_prompts(batch["text"], dataset_name=args.dataset)
            labels = batch["labels"].to(device).float()
            Hq = llama.encode(
                prompts, max_length=args.max_length, already_prompted=True
            ).float()
            Qq = QueryComp(Wq(Hq))
            Mem_flat = base_mem_flat.expand(Qq.size(0), -1, -1)
            z_hat, _ = Inference(Qq, Mem_flat)
            if standardize:
                y_hat = z_hat.squeeze(-1) * y_std + y_mean
            else:
                y_hat = z_hat.squeeze(-1)
            preds_all.append(y_hat.cpu())
            labels_all.append(labels.cpu())
        preds_cat = torch.cat(preds_all)
        labels_cat = torch.cat(labels_all)
        metrics_local = regression_metrics(preds_cat, labels_cat)
        return preds_cat, labels_cat, metrics_local

    preds_cat, labels_cat, metrics = run_eval(test_loader, "test")
    metrics_path = os.path.join(args.save_dir, args.metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics to {os.path.abspath(metrics_path)}")
    print(
        "Metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v == v)
    )

    if args.run_val and val_loader is not None:
        _, _, metrics_val = run_eval(val_loader, "val")
        val_path = os.path.join(args.save_dir, "metrics_val.json")
        with open(val_path, "w") as f:
            json.dump(metrics_val, f, indent=2)
        print(
            f"Saved val metrics to {os.path.abspath(val_path)} "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in metrics_val.items() if v == v
            )
        )

    # Explanations: sample 5 random test examples
    test_ds = dm.dataset["test"]
    sample_idxs = random.sample(range(len(test_ds)), k=min(5, len(test_ds)))
    explanations = []
    for idx_ds in sample_idxs:
        row = test_ds[idx_ds]
        raw_text = row["text"]
        prompts = prepare_prompts([raw_text], dataset_name=args.dataset)
        labels = torch.tensor(
            [float(row.get("labels", row.get("score", 0.0)))]
        ).to(device)
        Hq = llama.encode(
            prompts, max_length=args.max_length, already_prompted=True
        ).float()
        Qq = QueryComp(Wq(Hq))
        z_hat, attn = Inference(Qq, base_mem_flat)
        if standardize:
            y_hat = z_hat.squeeze(-1) * y_std + y_mean
        else:
            y_hat = z_hat.squeeze(-1)
        attn_slots = (
            attn.view(1, 1, idx.numel(), m + 1)
            .sum(dim=-1)
            .squeeze(0)
            .squeeze(0)
        )
        sorted_vals, sorted_slots = torch.sort(attn_slots, descending=True)
        proto_list = []
        for val, proto_slot in zip(sorted_vals.tolist(), sorted_slots.tolist()):
            data_idx = idx[proto_slot].item()
            proto_list.append(
                {
                    "data_idx": int(data_idx),
                    "attn": float(val),
                    "y": float(y_cache[data_idx].item()),
                    "text": text_cache[data_idx],
                }
            )
        explanations.append(
            {
                "test_index": int(idx_ds),
                "input_text": raw_text,
                "label": float(labels.cpu().item()),
                "prediction": float(y_hat.item()),
                "prototypes": proto_list,
            }
        )

    exp_path = os.path.join(args.save_dir, args.explanations_name)
    with open(exp_path, "w") as f:
        json.dump(explanations, f, ensure_ascii=True, indent=2)
    print(f"Saved sampled explanations to {os.path.abspath(exp_path)}")


if __name__ == "__main__":
    main()
