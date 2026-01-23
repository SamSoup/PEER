import argparse
import os
import sys
import random

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
)
from peer.utils import huber_loss, regression_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        description="Stage C fine-tune after prototype snapping (fixed prototypes)."
    )
    parser.add_argument("--model_name", required=True, help="HF LLaMA name.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--ckpt", default="stageB.pt", help="Path to Stage B checkpoint."
    )
    parser.add_argument(
        "--prototypes",
        default="prototypes.pt",
        help="Prototype indices tensor from finalize_prototypes.",
    )
    parser.add_argument(
        "--cache_dir",
        default="cache",
        help="Directory with cache_mem.pt/cache_y.pt/cache_text.json.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Stage C epochs (default 2)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate (AdamW)."
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
        "--save_dir",
        default=".",
        help="Directory to save Stage C checkpoint.",
    )
    parser.add_argument(
        "--ckpt_name",
        default="stageC.pt",
        help="Filename for Stage C checkpoint inside save_dir.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
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

    ckpt_b = torch.load(args.ckpt, map_location="cpu")
    if bool(ckpt_b.get("standardize_labels", True)) != bool(
        args.standardize_labels
    ):
        raise ValueError(
            "standardize_labels mismatch: Stage B checkpoint "
            f"{ckpt_b.get('standardize_labels', True)} vs Stage C arg {args.standardize_labels}"
        )
    Wq, QueryComp, LabelEmb, Inference, d_h = load_stage_b_modules(
        ckpt_b, device
    )

    mem_cache = torch.load(
        os.path.join(args.cache_dir, "cache_mem.pt"), map_location="cpu"
    ).to(device)
    y_cache = torch.load(
        os.path.join(args.cache_dir, "cache_y.pt"), map_location="cpu"
    ).to(device)
    prototypes = torch.load(args.prototypes, map_location="cpu").long()

    llama = FrozenLlama(args.model_name, device)
    train_loader, val_loader, _ = build_raw_dataloaders(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        combine_fields=False,
    )

    # label stats sanity
    train_labels = torch.tensor(
        train_loader.dataset["labels"], dtype=torch.float32
    )
    print(
        f"[StageC] label stats train mean={train_labels.mean().item():.4f} std={train_labels.std().item():.4f} "
        f"min={train_labels.min().item():.4f} max={train_labels.max().item():.4f}"
    )

    params = (
        list(Wq.parameters())
        + list(QueryComp.parameters())
        + list(LabelEmb.parameters())
        + list(Inference.parameters())
    )
    optim = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    # Fixed prototype memories (label embed recomputed each forward)
    Mem_sel = mem_cache[prototypes].float()  # [K, m, d_h]
    y_proto = y_cache[prototypes]  # [K]
    K = prototypes.numel()
    m = Mem_sel.size(1)

    best_metric_name = args.best_metric
    minimize = best_metric_name in ["val_loss", "mse", "rmse"]
    best_score = float("inf") if minimize else -float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_ckpt = None
    best_path = None

    for epoch in range(args.epochs):
        Wq.train(), QueryComp.train(), LabelEmb.train(), Inference.train()
        total_loss = 0.0
        n_steps = 0
        for batch in tqdm(
            train_loader,
            desc=f"StageC Epoch {epoch+1}/{args.epochs}",
            leave=False,
        ):
            prompts = prepare_prompts(batch["text"], dataset_name=args.dataset)
            labels = batch["labels"].to(device).float()

            Hq = llama.encode(
                prompts, max_length=args.max_length, already_prompted=True
            ).float()
            Qq = QueryComp(Wq(Hq))

            Ltok = LabelEmb(y_proto).squeeze(1)  # [K, d_h]
            Mem_slot = torch.cat(
                [Mem_sel, Ltok.unsqueeze(1)], dim=1
            )  # [K, m+1, d_h]
            Mem_flat = Mem_slot.view(1, K * (m + 1), d_h).expand(
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

            optim.zero_grad()
            L_reg.backward()
            if n_steps == 0:

                def gn(p):
                    return None if p.grad is None else p.grad.data.norm().item()

                gWq = gn(Wq.weight)
                gR0 = gn(Inference.R0)
                gOut = gn(Inference.out[0].weight)

                def fmt(val):
                    return f"{val:.4e}" if val is not None else "None"

                print(
                    "[StageC] grad norms: "
                    f"Wq={fmt(gWq)} "
                    f"r0={fmt(gR0)} "
                    f"out={fmt(gOut)}"
                )

            optim.step()
            total_loss += L_reg.item()
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)

        val_loss = None
        val_metrics = None
        if val_loader is not None:
            Wq.eval(), QueryComp.eval(), LabelEmb.eval(), Inference.eval()
            with torch.no_grad():
                v_total = 0.0
                v_steps = 0
                preds_all = []
                labels_all = []
                for batch in tqdm(val_loader, desc="StageC Val", leave=False):
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
                    Ltok = LabelEmb(y_proto).squeeze(1)
                    Mem_slot = torch.cat([Mem_sel, Ltok.unsqueeze(1)], dim=1)
                    Mem_flat = Mem_slot.view(1, K * (m + 1), d_h).expand(
                        Qq.size(0), -1, -1
                    )
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
                    else:
                        z_hat, _ = Inference(Qq, Mem_flat)
                        pred = z_hat.squeeze(-1)
                        v_total += huber_loss(
                            pred, labels, delta=args.huber_delta
                        ).item()
                    preds_all.append(pred.cpu())
                    labels_all.append(labels.cpu())
                    v_steps += 1
                val_loss = v_total / max(v_steps, 1)
                if preds_all:
                    preds_cat = torch.cat(preds_all)
                    labels_cat = torch.cat(labels_all)
                    val_metrics = regression_metrics(preds_cat, labels_cat)
                    y_min, y_max = (
                        labels_cat.min().item(),
                        labels_cat.max().item(),
                    )
                    denom = max(y_max - y_min, 1e-6)
                    sat_rate = (
                        (preds_cat < y_min + 0.01 * denom)
                        | (preds_cat > y_max - 0.01 * denom)
                    ).float().mean().item() * 100.0
                    stats_msg = (
                        f"val std(yhat)={preds_cat.std().item():.4f} "
                        f"min/max(yhat)={[preds_cat.min().item(), preds_cat.max().item()]} "
                        f"mean(yhat)={preds_cat.mean().item():.4f} "
                        f"sat%={sat_rate:.2f}"
                    )
                    print(stats_msg)
                    current_score = None
                    if best_metric_name == "val_loss":
                        current_score = val_loss
                    else:
                        current_score = val_metrics.get(best_metric_name)

                    better = False
                    if (
                        current_score is not None
                        and current_score == current_score
                    ):
                        if minimize:
                            better = current_score < best_score
                        else:
                            better = current_score > best_score

                    if better:
                        best_score = current_score
                        best_epoch = epoch + 1
                        best_val_metrics = val_metrics
                        best_ckpt = {
                            "model_name": args.model_name,
                            "d_model": ckpt_b["d_model"],
                            "d_h": d_h,
                            "Wq": Wq.state_dict(),
                            "QueryComp": QueryComp.state_dict(),
                            "LabelEmb": LabelEmb.state_dict(),
                            "InferenceHead": Inference.state_dict(),
                            "label_mean": ckpt_b.get("label_mean", 0.0),
                            "label_std": ckpt_b.get("label_std", 1.0),
                            "K": K,
                            "m": m,
                            "standardize_labels": args.standardize_labels,
                            "best_metric": best_metric_name,
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                            "best_val_metrics": best_val_metrics,
                        }
                        best_path = os.path.join(args.save_dir, args.ckpt_name)
                        torch.save(best_ckpt, best_path)
                        print(
                            f"Saved new best Stage C to {os.path.abspath(best_path)} ({best_metric_name}={best_score:.4f}, epoch={best_epoch})"
                        )

        print(
            f"Epoch {epoch+1}/{args.epochs} - train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
        )
        if val_metrics is not None:
            print(
                "Val metrics: "
                + ", ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items() if v == v
                )
            )

    ckpt_out = best_ckpt
    if ckpt_out is None:
        ckpt_out = {
            "model_name": args.model_name,
            "d_model": ckpt_b["d_model"],
            "d_h": d_h,
            "Wq": Wq.state_dict(),
            "QueryComp": QueryComp.state_dict(),
            "LabelEmb": LabelEmb.state_dict(),
            "InferenceHead": Inference.state_dict(),
            "label_mean": ckpt_b.get("label_mean", 0.0),
            "label_std": ckpt_b.get("label_std", 1.0),
            "K": K,
            "m": m,
            "standardize_labels": args.standardize_labels,
            "best_metric": best_metric_name,
            "best_score": (
                best_score
                if best_score
                not in (float("inf") if minimize else -float("inf"))
                else None
            ),
            "best_epoch": best_epoch if best_epoch >= 0 else None,
            "best_val_metrics": best_val_metrics,
        }
        out_path = os.path.join(args.save_dir, args.ckpt_name)
        torch.save(ckpt_out, out_path)
        print(
            f"Saved Stage C checkpoint to {os.path.abspath(out_path)}; best_epoch={best_epoch}, best_{best_metric_name}={best_score}"
        )
        ckpt_verify = torch.load(out_path, map_location="cpu")
        print(
            f"[StageC] reload check -> best_epoch={ckpt_verify.get('best_epoch')}, "
            f"best_score={ckpt_verify.get('best_score')}, "
            f"best_metric={ckpt_verify.get('best_metric')}"
        )
    else:
        print(
            f"Best Stage C checkpoint already saved at {os.path.abspath(best_path) if best_path else args.ckpt_name} "
            f"(best_epoch={best_epoch}, best_{best_metric_name}={best_score})"
        )


if __name__ == "__main__":
    main()
