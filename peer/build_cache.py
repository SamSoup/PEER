import argparse
import json
import os
import sys
import hashlib

import torch
import torch.nn as nn
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modelsv2.data import build_raw_dataloaders, build_cache_loader, prepare_prompts
from modelsv2.llama_backbone import FrozenLlama
from modelsv2.modules import PerceiverCompressor, SlotSelector, KeyReadout


def hash_state_dict(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if torch.is_tensor(v):
            h.update(v.cpu().numpy().tobytes())
        else:
            h.update(str(v).encode("utf-8"))
    return h.hexdigest()


def load_stage_a_modules(ckpt, device):
    d_model = ckpt["d_model"]
    d_h = ckpt.get("d_h", 256)
    m = ckpt.get("m", 8)
    n_heads = ckpt.get("n_heads", 8)
    Wm = nn.Linear(d_model, d_h).to(device)
    MemComp = PerceiverCompressor(d_h=d_h, m=m, n_heads=n_heads).to(device)
    KeyRO = KeyReadout(d_h=d_h, n_heads=n_heads).to(device)
    Wm.load_state_dict(ckpt["Wm"])
    MemComp.load_state_dict(ckpt["MemComp"])
    KeyRO.load_state_dict(ckpt["KeyReadout"])
    Wm.eval()
    MemComp.eval()
    KeyRO.eval()
    return Wm, MemComp, KeyRO


def main():
    parser = argparse.ArgumentParser(
        description="Build compressed memory cache from Stage A model."
    )
    parser.add_argument("--model_name", required=True, help="HF LLaMA name.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--ckpt", default="stageA.pt", help="Path to Stage A checkpoint."
    )
    parser.add_argument(
        "--cache_dir", default="cache", help="Output directory for cache files."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for caching."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Prompt tokenization max length.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.cache_dir, exist_ok=True)

    def _to_display(t):
        if isinstance(t, (list, tuple)) and len(t) == 2:
            return f"{t[0]} [SEP] {t[1]}"
        return str(t)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    Wm, MemComp, KeyRO = load_stage_a_modules(ckpt, device)

    llama = FrozenLlama(args.model_name, device)
    _, _, dm = build_raw_dataloaders(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        combine_fields=False,
    )
    cache_loader = build_cache_loader(dm, args.batch_size)

    mem_list = []
    key_list = []
    y_list = []
    text_list = []

    for batch in tqdm(cache_loader, desc="Building cache", leave=False):
        prompts = prepare_prompts(batch["text"], dataset_name=args.dataset)
        labels = batch["labels"].to(device).float()
        H = llama.encode(
            prompts, max_length=args.max_length, already_prompted=True
        ).float()
        with torch.no_grad():
            M = MemComp(Wm(H))
            k_read = KeyRO(M)
        mem_list.append(M.detach().to(torch.float16).cpu())
        keys = SlotSelector.l2norm(k_read).to(torch.float16).cpu()
        key_list.append(keys)
        y_list.append(labels.cpu())
        text_list.extend([_to_display(t) for t in batch["text"]])

    mem_cache = torch.cat(mem_list, dim=0)
    key_cache = torch.cat(key_list, dim=0)
    y_cache = torch.cat(y_list, dim=0).float()

    torch.save(mem_cache, os.path.join(args.cache_dir, "cache_mem.pt"))
    torch.save(key_cache, os.path.join(args.cache_dir, "cache_keys.pt"))
    torch.save(y_cache, os.path.join(args.cache_dir, "cache_y.pt"))
    with open(os.path.join(args.cache_dir, "cache_text.json"), "w") as f:
        json.dump(text_list, f, ensure_ascii=True, indent=2)

    meta = {
        "stageA_ckpt": os.path.abspath(args.ckpt),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "label_mean": ckpt.get("label_mean", None),
        "label_std": ckpt.get("label_std", None),
        "standardize_labels": ckpt.get("standardize_labels", None),
        "huber_delta": ckpt.get("huber_delta", None),
        "Wm_hash": hash_state_dict(ckpt["Wm"]),
        "MemComp_hash": hash_state_dict(ckpt["MemComp"]),
        "KeyReadout_hash": hash_state_dict(ckpt["KeyReadout"]),
    }
    with open(os.path.join(args.cache_dir, "cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Cached {mem_cache.size(0)} examples to {os.path.abspath(args.cache_dir)}"
    )


if __name__ == "__main__":
    main()
