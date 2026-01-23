import argparse
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from peer.modules import SlotSelector


def greedy_dedup(logits_full: torch.Tensor):
    """
    Greedy no-dup selection per slot: take best-scoring unused index.
    """
    K, N = logits_full.shape
    selected = []
    used = set()
    for k in range(K):
        scores, idxs = torch.sort(logits_full[k], descending=True)
        pick = None
        for idx in idxs:
            i = idx.item()
            if i not in used:
                pick = i
                break
        if pick is None:
            pick = idxs[0].item()
        selected.append(pick)
        used.add(pick)
    return torch.tensor(selected, dtype=torch.long)


def shortlist_candidates(logits_full: torch.Tensor, topk: int):
    """Union of per-slot top-k indices."""
    K, N = logits_full.shape
    k = min(max(topk, 1), N)
    _, top_idx = torch.topk(logits_full, k=k, dim=1)
    cand = torch.unique(top_idx.reshape(-1))
    return cand


def greedy_on_shortlist(logits_full: torch.Tensor, cand: torch.Tensor):
    sub = logits_full[:, cand]  # [K, M]
    picked_sub = greedy_dedup(sub)
    return cand[picked_sub]


def global_assignment(logits_full: torch.Tensor, cand: torch.Tensor):
    """
    Attempt global max-sum assignment over shortlisted candidates.
    Falls back to greedy with random restarts if SciPy is unavailable.
    """
    K, _ = logits_full.shape
    M = cand.numel()
    if M < K:
        return None
    sub = logits_full[:, cand]  # [K, M]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment((-sub).cpu().numpy())
        # Expect row_ind to cover 0..K-1
        picked = cand[col_ind]
        return picked
    except Exception:
        # Fallback: random-restart greedy on shortlist
        import random

        best = None
        best_score = -1e18
        order = list(range(K))
        sub_cpu = sub.cpu()
        for _ in range(50):
            random.shuffle(order)
            used = set()
            picks = []
            for k in order:
                scores, idxs = torch.sort(sub_cpu[k], descending=True)
                pick = None
                for idx in idxs:
                    i = idx.item()
                    if i not in used:
                        pick = i
                        used.add(i)
                        break
                if pick is None:
                    pick = idxs[0].item()
                    used.add(pick)
                picks.append(pick)
            # restore original slot order
            restored = [None] * K
            for slot, cand_idx in zip(order, picks):
                restored[slot] = cand_idx
            pick_tensor = torch.tensor(restored, dtype=torch.long)
            score = sub_cpu[torch.arange(K), pick_tensor].sum().item()
            if score > best_score:
                best_score = score
                best = pick_tensor
        return cand[best] if best is not None else None


def main():
    parser = argparse.ArgumentParser(
        description="Finalize hard prototype indices from SlotSelector."
    )
    parser.add_argument(
        "--ckpt", default="stageB.pt", help="Path to Stage B checkpoint."
    )
    parser.add_argument(
        "--cache_dir",
        default="cache",
        help="Directory containing cache_keys.pt and cache_text.json.",
    )
    parser.add_argument(
        "--save_dir",
        default=".",
        help="Directory to save prototypes.",
    )
    parser.add_argument(
        "--pt_name",
        default="prototypes.pt",
        help="Filename for prototype indices tensor.",
    )
    parser.add_argument(
        "--json_name",
        default="prototypes.json",
        help="Filename for human-readable prototype info.",
    )
    parser.add_argument(
        "--assign_mode",
        choices=["greedy", "global"],
        default="greedy",
        help="Assignment strategy over shortlisted candidates.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Per-slot top-k to form candidate pool (defaults to Stage B T).",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    selector_state = ckpt["SlotSelector"]
    slot_q = selector_state["slot_q"]
    K = slot_q.shape[0]
    d_h = slot_q.shape[1]
    selector = SlotSelector(K=K, d_h=d_h, T=selector_state.get("T", 512))
    selector.load_state_dict(selector_state)

    keys = torch.load(
        os.path.join(args.cache_dir, "cache_keys.pt"), map_location="cpu"
    )
    y_cache = torch.load(
        os.path.join(args.cache_dir, "cache_y.pt"), map_location="cpu"
    )
    with open(os.path.join(args.cache_dir, "cache_text.json"), "r") as f:
        text_cache = json.load(f)

    keys_n = SlotSelector.l2norm(keys.float())
    slot_n = SlotSelector.l2norm(selector.slot_q.float())
    logits_full = slot_n @ keys_n.T  # [K, N]

    top_default = ckpt.get("T", 512)
    topk = args.topk if args.topk is not None else top_default
    candidates = shortlist_candidates(logits_full, topk)
    if candidates.numel() < K:
        print(
            f"[WARN] shortlist ({candidates.numel()}) smaller than K={K}; falling back to full greedy."
        )
        prototypes = greedy_dedup(logits_full)
    else:
        if args.assign_mode == "global":
            picked = global_assignment(logits_full, candidates)
            if picked is None:
                prototypes = greedy_on_shortlist(logits_full, candidates)
            else:
                prototypes = picked
        else:
            prototypes = greedy_on_shortlist(logits_full, candidates)

    out_pt = os.path.join(args.save_dir, args.pt_name)
    out_json = os.path.join(args.save_dir, args.json_name)
    torch.save(prototypes, out_pt)

    proto_info = []
    for idx in prototypes.tolist():
        proto_info.append(
            {
                "idx": idx,
                "y": float(y_cache[idx].item()),
                "text": text_cache[idx],
            }
        )
    with open(out_json, "w") as f:
        json.dump(proto_info, f, ensure_ascii=True, indent=2)

    print(f"Saved prototype indices to {os.path.abspath(out_pt)}")
    print(f"Saved prototype details to {os.path.abspath(out_json)}")


if __name__ == "__main__":
    main()
