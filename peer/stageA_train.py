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
    KeyReadout,
)
from peer.utils import (
    huber_loss,
    regression_metrics,
    set_label_stats_from_loader,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_stage_a(
    model_name: str,
    dataset: str,
    device: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    max_length: int = 256,
    d_h: int = 256,
    m: int = 8,
    mq: int = 8,
    n_heads: int = 8,
    L: int = 3,
    lambda_emb: float = 0.1,
    huber_delta: float = 1.0,
    standardize_labels: bool = True,
    seed: int = 0,
    save_dir: str = ".",
    ckpt_name: str = "stageA.pt",
    best_metric: str = "val_loss",
):
    set_seed(seed)
    llama = FrozenLlama(model_name, device)
    d_model = llama.model.config.hidden_size

    train_loader, val_loader, dm = build_raw_dataloaders(
        dataset_name=dataset,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        combine_fields=False,
    )

    Wm = nn.Linear(d_model, d_h).to(device)
    Wq = nn.Linear(d_model, d_h).to(device)
    MemComp = PerceiverCompressor(d_h=d_h, m=m, n_heads=n_heads).to(device)
    QueryComp = PerceiverCompressor(d_h=d_h, m=mq, n_heads=n_heads).to(device)
    LabelEmb = ScalarLabelEmbedder(d_h=d_h).to(device)
    Inference = InferenceHead(d_h=d_h, n_heads=n_heads, L=L).to(device)
    KeyRO = KeyReadout(d_h=d_h, n_heads=n_heads).to(device)
    g_emb = nn.Sequential(
        nn.Linear(d_h, d_h),
        nn.GELU(),
        nn.Linear(d_h, d_h),
    ).to(device)

    label_mean, label_std = set_label_stats_from_loader(LabelEmb, train_loader)
    # Sanity on label stats
    print(
        f"[StageA] label stats train mean={label_mean:.4f} std={label_std:.4f} "
        f"min={min(train_loader.dataset['labels']):.4f} max={max(train_loader.dataset['labels']):.4f}"
        if hasattr(train_loader, "dataset")
        else f"[StageA] label stats train mean={label_mean:.4f} std={label_std:.4f}"
    )

    params = (
        list(Wm.parameters())
        + list(Wq.parameters())
        + list(MemComp.parameters())
        + list(QueryComp.parameters())
        + list(LabelEmb.parameters())
        + list(Inference.parameters())
        + list(KeyRO.parameters())
        + list(g_emb.parameters())
    )
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best_metric_name = best_metric
    minimize = best_metric_name in ["val_loss", "mse", "rmse"]
    best_score = float("inf") if minimize else -float("inf")
    best_ckpt = None
    best_path = None

    # Quick input/tokenization sanity on first batch
    first_batch = next(iter(train_loader))
    sample_texts = (
        first_batch["text"]
        if "text" in first_batch
        else first_batch.get("input_text", [])
    )
    print(
        "[StageA] SAMPLE TEXTS:\n"
        + "\n---\n".join([str(t) for t in sample_texts[:3]])
    )
    tok_dbg = llama.tok(
        sample_texts[:8],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    print(
        "[StageA] token lengths:", tok_dbg["attention_mask"].sum(dim=1).tolist()
    )
    print(
        "[StageA] unique input_ids per example:",
        [ids.unique().numel() for ids in tok_dbg["input_ids"]],
    )
    if tok_dbg["input_ids"].size(0) >= 2:
        eq01 = torch.equal(tok_dbg["input_ids"][0], tok_dbg["input_ids"][1])
        print("[StageA] example0 == example1 input_ids?", eq01)

    for epoch in range(epochs):
        Wm.train(), Wq.train(), MemComp.train(), QueryComp.train()
        LabelEmb.train(), Inference.train()
        total_loss = 0.0
        n_steps = 0
        for batch in tqdm(
            train_loader, desc=f"StageA Epoch {epoch+1}/{epochs}", leave=False
        ):
            prompts = prepare_prompts(batch["text"], dataset_name=dataset)
            labels = batch["labels"].to(device).float()

            H = llama.encode(
                prompts, max_length=max_length, already_prompted=True
            ).float()
            Tq = Wq(H)
            Qq = QueryComp(Tq)
            Tm = Wm(H)
            M = MemComp(Tm)
            if standardize_labels:
                y_std = label_std + 1e-8
                z_target = (labels - label_mean) / y_std
                z_hat, _ = Inference(Qq, M)
                base_loss = huber_loss(
                    z_hat.squeeze(-1), z_target, delta=huber_delta
                )
            else:
                z_hat, _ = Inference(Qq, M)
                base_loss = huber_loss(
                    z_hat.squeeze(-1), labels, delta=huber_delta
                )

            # auxiliary embedding prediction
            k_read = KeyRO(M)  # [B,d_h]
            e_target = LabelEmb(labels).squeeze(1)  # [B,d_h]
            e_pred = g_emb(k_read)
            L_emb = torch.mean((e_pred - e_target) ** 2)

            loss = base_loss + lambda_emb * L_emb

            optim.zero_grad()
            loss.backward()
            # gradient sanity (first batch per epoch)
            if n_steps == 0:

                def gn(param):
                    if param.grad is None:
                        return None
                    return param.grad.data.norm().item()

                gWq = gn(Wq.weight)
                gWm = gn(Wm.weight)
                gR0 = gn(Inference.R0)
                gOut = gn(Inference.out[0].weight)

                def fmt(val):
                    return f"{val:.4e}" if val is not None else "None"

                print(
                    "[StageA] grad norms: "
                    f"Wq={fmt(gWq)} "
                    f"Wm={fmt(gWm)} "
                    f"r0={fmt(gR0)} "
                    f"out={fmt(gOut)}"
                )
            optim.step()

            total_loss += loss.item()
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)

        val_loss = None
        metrics = None
        if val_loader is not None:
            Wm.eval(), Wq.eval(), MemComp.eval(), QueryComp.eval()
            LabelEmb.eval(), Inference.eval()
            KeyRO.eval(), g_emb.eval()
            with torch.no_grad():
                v_total = 0.0
                v_steps = 0
                preds_all = []
                labels_all = []
                for batch in tqdm(val_loader, desc="StageA Val", leave=False):
                    prompts = prepare_prompts(
                        batch["text"], dataset_name=dataset
                    )
                    labels = batch["labels"].to(device).float()
                    H = llama.encode(
                        prompts, max_length=max_length, already_prompted=True
                    ).float()
                    Qq = QueryComp(Wq(H))
                    M = MemComp(Wm(H))
                    if standardize_labels:
                        y_std = label_std + 1e-8
                        z_target = (labels - label_mean) / y_std
                        z_hat, _ = Inference(Qq, M)
                        pred = z_hat.squeeze(-1) * y_std + label_mean
                        v_total += huber_loss(
                            z_hat.squeeze(-1), z_target, delta=huber_delta
                        ).item()
                    else:
                        z_hat, _ = Inference(Qq, M)
                        pred = z_hat.squeeze(-1)
                        v_total += huber_loss(
                            pred, labels, delta=huber_delta
                        ).item()
                    preds_all.append(pred.cpu())
                    labels_all.append(labels.cpu())
                    v_steps += 1
                val_loss = v_total / max(v_steps, 1)
                if preds_all:
                    preds_cat = torch.cat(preds_all)
                    labels_cat = torch.cat(labels_all)
                    metrics = regression_metrics(preds_cat, labels_cat)
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
                else:
                    metrics = None
                current_score = None
                if best_metric_name == "val_loss":
                    current_score = val_loss
                elif metrics is not None:
                    current_score = metrics.get(best_metric_name)

                better = False
                if (
                    current_score is not None and current_score == current_score
                ):  # not NaN
                    if minimize:
                        better = current_score < best_score
                    else:
                        better = current_score > best_score

                if better:
                    best_score = current_score
                    best_ckpt = {
                        "model_name": model_name,
                        "d_model": d_model,
                        "d_h": d_h,
                        "m": m,
                        "mq": mq,
                        "n_heads": n_heads,
                        "L": L,
                        "lambda_emb": lambda_emb,
                        "standardize_labels": standardize_labels,
                        "huber_delta": huber_delta,
                        "Wm": Wm.state_dict(),
                        "Wq": Wq.state_dict(),
                        "MemComp": MemComp.state_dict(),
                        "QueryComp": QueryComp.state_dict(),
                        "LabelEmb": LabelEmb.state_dict(),
                        "InferenceHead": Inference.state_dict(),
                        "KeyReadout": KeyRO.state_dict(),
                        "g_emb": g_emb.state_dict(),
                        "label_mean": label_mean,
                        "label_std": label_std,
                        "best_metric": best_metric_name,
                        "best_score": best_score,
                    }
                    best_path = os.path.join(save_dir, ckpt_name)
                    torch.save(best_ckpt, best_path)
                    print(
                        f"Saved new best Stage A to {os.path.abspath(best_path)} ({best_metric_name}={best_score:.4f})"
                    )

        print(
            f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
        )
        if metrics is not None:
            print(
                "Val metrics: "
                + ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in metrics.items()
                    if v == v  # skip NaN
                )
            )

    if best_ckpt is None:
        ckpt = {
            "model_name": model_name,
            "d_model": d_model,
            "d_h": d_h,
            "m": m,
            "mq": mq,
            "n_heads": n_heads,
            "L": L,
            "lambda_emb": lambda_emb,
            "standardize_labels": standardize_labels,
            "huber_delta": huber_delta,
            "Wm": Wm.state_dict(),
            "Wq": Wq.state_dict(),
            "MemComp": MemComp.state_dict(),
            "QueryComp": QueryComp.state_dict(),
            "LabelEmb": LabelEmb.state_dict(),
            "InferenceHead": Inference.state_dict(),
            "KeyReadout": KeyRO.state_dict(),
            "g_emb": g_emb.state_dict(),
            "label_mean": label_mean,
            "label_std": label_std,
            "best_metric": best_metric_name,
            "best_score": (
                best_score
                if best_score != (float("inf") if minimize else -float("inf"))
                else None
            ),
        }
        ckpt_path = os.path.join(save_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)
        print(
            f"Saved Stage A checkpoint to {os.path.abspath(ckpt_path)} (no val improvement over baseline)"
        )
    else:
        print(
            f"Best Stage A checkpoint already saved at {os.path.abspath(best_path) if best_path else ckpt_name}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stage A training for prototype regressor."
    )
    parser.add_argument("--model_name", required=True, help="HF LLaMA name.")
    parser.add_argument(
        "--dataset", required=True, help="Dataset name from /data."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Stage A epochs (default 5)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (AdamW)."
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
        "--d_h", type=int, default=256, help="Head dimension d_h."
    )
    parser.add_argument(
        "--m", type=int, default=8, help="Memory tokens per example."
    )
    parser.add_argument(
        "--mq", type=int, default=8, help="Compressed query tokens."
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Attention heads in compressors/head.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Inference head layers."
    )
    parser.add_argument(
        "--lambda_emb",
        type=float,
        default=0.1,
        help="Weight for auxiliary embedding loss.",
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
        "--save_dir",
        default=".",
        help="Directory to save Stage A checkpoint.",
    )
    parser.add_argument(
        "--ckpt_name",
        default="stageA.pt",
        help="Filename for Stage A checkpoint inside save_dir.",
    )
    parser.add_argument(
        "--best_metric",
        choices=["val_loss", "mse", "rmse", "pearson", "spearman", "kendall"],
        default="val_loss",
        help="Metric to select best checkpoint (min for val_loss/mse/rmse, max for correlations).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    train_stage_a(
        model_name=args.model_name,
        dataset=args.dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        d_h=args.d_h,
        m=args.m,
        mq=args.mq,
        n_heads=args.n_heads,
        L=args.num_layers,
        lambda_emb=args.lambda_emb,
        huber_delta=args.huber_delta,
        standardize_labels=args.standardize_labels,
        seed=args.seed,
        save_dir=args.save_dir,
        ckpt_name=args.ckpt_name,
        best_metric=args.best_metric,
    )


if __name__ == "__main__":
    main()
