import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from tqdm.auto import tqdm

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
    if standardize:
        print("Is standardized!")

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
        texts_all = []
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            prompts = prepare_prompts(batch["text"], dataset_name=args.dataset)
            labels = batch["labels"].to(device).float()
            texts_all.extend(list(batch["text"]))
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
        return preds_cat, labels_cat, metrics_local, texts_all

    def contribution_importance_to_prototypes(
        contrib_layers,
        grad_r,
        n_proto,
        tokens_per_proto,
        batch_size,
    ):
        n_mem = n_proto * tokens_per_proto
        if (
            not isinstance(contrib_layers, (list, tuple))
            or len(contrib_layers) == 0
        ):
            raise ValueError("Expected non-empty per-layer contribution list.")

        def normalize_to_btne(contrib_one):
            if contrib_one.dim() != 4:
                raise ValueError(
                    "Expected contribution tensor rank-4 [B,T,N,E], got "
                    f"{tuple(contrib_one.shape)}"
                )
            if contrib_one.shape[0] == batch_size:
                btne = contrib_one
            elif contrib_one.shape[1] == batch_size:
                btne = contrib_one.transpose(0, 1)
            else:
                raise ValueError(
                    "Could not identify batch dimension in contribution shape "
                    f"{tuple(contrib_one.shape)}"
                )
            if btne.shape[2] != n_mem:
                raise ValueError(
                    f"Unexpected contribution shape {tuple(contrib_one.shape)} "
                    f"for n_mem={n_mem}"
                )
            return btne

        layer_scores = []
        for contrib_l in contrib_layers:
            btne = normalize_to_btne(contrib_l)
            B, T, _, E = btne.shape
            proto_vec = btne.reshape(B, T, n_proto, tokens_per_proto, E).sum(
                dim=3
            )  # [B,T,K,E]
            # Faithful additive contribution across readout tokens.
            proto_vec = proto_vec.sum(dim=1)  # [B,K,E]
            layer_score = (proto_vec * grad_r.unsqueeze(1)).sum(dim=-1)  # [B,K]
            layer_scores.append(layer_score)

        # Total signed contribution across all inference layers.
        return torch.stack(layer_scores, dim=0).sum(dim=0)  # [B,K]

    preds_cat, labels_cat, metrics, eval_texts = run_eval(test_loader, "test")
    metrics_path = os.path.join(args.save_dir, args.metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics to {os.path.abspath(metrics_path)}")
    print(
        "Metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v == v)
    )

    if args.run_val and val_loader is not None:
        _, _, metrics_val, _ = run_eval(val_loader, "val")
        val_path = os.path.join(args.save_dir, "metrics_val.json")
        with open(val_path, "w") as f:
            json.dump(metrics_val, f, indent=2)
        print(
            f"Saved val metrics to {os.path.abspath(val_path)} "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in metrics_val.items() if v == v
            )
        )

    # Explanations: write all test examples (evaluation order)
    if len(eval_texts) != preds_cat.numel():
        raise ValueError(
            "Evaluated text count mismatch with predictions: "
            f"{len(eval_texts)} vs {preds_cat.numel()}"
        )

    squared_errors = (preds_cat - labels_cat).pow(2)
    n_test = squared_errors.numel()
    sample_idxs = list(range(n_test))

    explanations = []
    for idx_ds in tqdm(sample_idxs, desc="Building explanations"):
        raw_text = eval_texts[idx_ds]
        label_eval = float(labels_cat[idx_ds].item())
        pred_eval = float(preds_cat[idx_ds].item())
        mse_eval = float(squared_errors[idx_ds].item())
        prompts = prepare_prompts([raw_text], dataset_name=args.dataset)
        Hq = llama.encode(
            prompts, max_length=args.max_length, already_prompted=True
        ).float()
        Qq = QueryComp(Wq(Hq))
        Mem_flat = base_mem_flat.expand(Qq.size(0), -1, -1)
        _, r_final, _, contrib_layers = Inference.forward_with_r_mem_contrib(
            Qq, Mem_flat
        )

        # Local output-direction for first-order faithful scalar attribution.
        with torch.enable_grad():
            r_local = r_final.detach().requires_grad_(True)
            raw_local = Inference.out(r_local[:, 0, :]).squeeze(-1)
            if standardize:
                raw_local = raw_local * y_std + y_mean
            grad_r = torch.autograd.grad(raw_local.sum(), r_local)[0][
                :, 0, :
            ].detach()

        contrib_scores = contribution_importance_to_prototypes(
            contrib_layers,
            grad_r,
            idx.numel(),
            m + 1,
            batch_size=Qq.size(0),
        )
        contrib_slots = contrib_scores[0]
        eps = 1e-8
        total_abs = contrib_slots.abs().sum() + eps
        total_negative = contrib_slots[contrib_slots < 0].sum()
        total_positive = contrib_slots[contrib_slots > 0].sum()
        total_signed = total_negative + total_positive

        pct_abs_total = 100.0 * contrib_slots / total_abs
        pct_rel_pos_neg = torch.zeros_like(contrib_slots)
        pos_mask = contrib_slots > 0
        neg_mask = contrib_slots < 0
        if total_positive.abs().item() > 0:
            pct_rel_pos_neg[pos_mask] = (
                100.0 * contrib_slots[pos_mask] / total_positive
            )
        if total_negative.abs().item() > 0:
            pct_rel_pos_neg[neg_mask] = (
                100.0 * contrib_slots[neg_mask] / total_negative
            )

        sorted_slots = torch.argsort(pct_abs_total.abs(), descending=True)
        proto_list = []
        for proto_slot in sorted_slots.tolist():
            data_idx = idx[proto_slot].item()
            proto_list.append(
                {
                    "data_idx": int(data_idx),
                    "contrib": float(contrib_slots[proto_slot].item()),
                    "pct_abs_total_influence": float(
                        pct_abs_total[proto_slot].item()
                    ),
                    "pct_of_pos_or_neg_total": float(
                        pct_rel_pos_neg[proto_slot].item()
                    ),
                    "y": float(y_cache[data_idx].item()),
                    "text": text_cache[data_idx],
                }
            )
        explanations.append(
            {
                "test_index": int(idx_ds),
                "input_text": raw_text,
                "label": label_eval,
                "prediction": pred_eval,
                "mse": mse_eval,
                "total_negative_contrib": float(total_negative.item()),
                "total_positive_contrib": float(total_positive.item()),
                "total_signed_contrib": float(total_signed.item()),
                "prototypes": proto_list,
            }
        )

    exp_path = os.path.join(args.save_dir, args.explanations_name)
    with open(exp_path, "w") as f:
        json.dump(explanations, f, ensure_ascii=True, indent=2)
    print(f"Saved sampled explanations to {os.path.abspath(exp_path)}")


if __name__ == "__main__":
    main()
