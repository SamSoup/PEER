from pathlib import Path
from typing import Any, Iterable, List, Tuple
from tqdm import tqdm
from wrapperbox.SentenceEncoder import SentenceEncoder
from data.factory import build_pairwise_datamodule
from data.pair_dataset_registry import get_dataset_meta
from peer.utils import ensure_hf_cache
import argparse
import numpy as np
import torch


def _native_sep(tokenizer) -> str:
    # Prefer sep_token, fallback to eos_token, fallback to "\n\n"
    sep = getattr(tokenizer, "sep_token", None)
    if sep:
        return f" {sep} "
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        return f" {eos} "
    return "\n\n"


def _batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _maybe_build_prompts(
    texts: List[str], dataset_name: str, use_prompts: bool
) -> List[str]:
    if not use_prompts:
        return texts
    try:
        from peer.prompting import build_prompts
    except Exception:
        return texts
    return build_prompts(texts, dataset_name)


def _extract_texts_for_split(
    dm,
    split: str,
    sep: str,
    dataset_meta,
) -> List[str]:
    """
    dm is built with tokenize_inputs=False, so mapped dataset should have:
      - "text": str OR (str,str) depending on combine_fields
    We ensure List[str] output.
    """
    ds = dm.dataset[split]
    out: List[str] = []

    # If dm already combined fields into a string, "text" is str.
    # If not, may be tuple/list length 2; we combine here.
    for ex in ds:
        x = ex.get("text")
        if isinstance(x, str):
            out.append(x)
            continue
        if isinstance(x, (tuple, list)) and len(x) == 2:
            out.append(f"{str(x[0])}{sep}{str(x[1])}")
            continue

        # fallback: try raw fields if needed
        s1f = getattr(dataset_meta, "sentence1_field", None)
        s2f = getattr(dataset_meta, "sentence2_field", None)
        if s1f and s2f and s1f in ex and s2f in ex:
            out.append(f"{str(ex[s1f])}{sep}{str(ex[s2f])}")
        else:
            out.append(str(x))

    return out


def _write_embeddings_npy(
    encoder: SentenceEncoder,
    texts: List[str],
    out_path: Path,
    batch_size: int,
    desc: str = "embed",
) -> Tuple[int, int]:
    """
    Streams embeddings to .npy using open_memmap. Returns (N, D).
    """
    if len(texts) == 0:
        # write an empty (0,0) file for consistency
        arr = np.zeros((0, 0), dtype=np.float32)
        np.save(out_path, arr)
        return 0, 0

    # First batch to infer D
    first_batch = texts[: min(batch_size, len(texts))]
    with torch.no_grad():
        emb0 = (
            encoder.encode(first_batch)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )

    N = len(texts)
    D = int(emb0.shape[1])

    mmap = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(N, D)
    )

    # write first batch
    mmap[: emb0.shape[0], :] = emb0
    offset = emb0.shape[0]

    progress = tqdm(total=N, initial=offset, desc=desc, unit="txt")

    # rest
    for batch in _batched(texts[offset:], batch_size):
        if not batch:
            continue
        with torch.no_grad():
            embs = (
                encoder.encode(batch)
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
        bsz = embs.shape[0]
        mmap[offset : offset + bsz, :] = embs
        offset += bsz
        progress.update(bsz)

    mmap.flush()
    progress.close()
    return N, D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LLM embeddings for dataset splits"
    )

    p.add_argument(
        "--dataset", type=str, required=True, help="Registry dataset name"
    )
    p.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help="HF model name for embeddings",
    )
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Token truncation length",
    )
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--normalize_embeddings", action="store_true")

    # prompt handling
    p.add_argument(
        "--no_prompts",
        action="store_true",
        help="Do not wrap texts into prompts",
    )

    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or ensure_hf_cache()

    # Build encoder first so we can choose tokenizer-native separator
    encoder = SentenceEncoder(
        model_name=args.llm_name,
        cache_dir=cache_dir,
        normalize_embeddings=args.normalize_embeddings,
        batch_size=args.batch_size,
        max_tokens=args.max_seq_length,
    )
    sep = _native_sep(encoder.tokenizer)

    # Determine if dataset is pairwise (by meta)
    meta = get_dataset_meta(args.dataset)
    if meta is None:
        raise ValueError(f"Unsupported dataset name: {args.dataset}")

    is_pairwise = bool(getattr(meta, "sentence2_field", None))

    # Build RAW datamodule; if pairwise, combine_fields=True so dm emits strings
    dm = build_pairwise_datamodule(
        dataset_name=args.dataset,
        model_name=args.llm_name,  # only used to init tokenizer in dm base, ok
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        tokenize_inputs=False,
        combine_fields=bool(is_pairwise),
        combine_separator_token=(
            getattr(encoder.tokenizer, "sep_token", None)
            or getattr(encoder.tokenizer, "eos_token", None)
            or "[SEP]"
        ),
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset name: {args.dataset}")

    dm.setup(stage=None)

    # Splits to try
    split_map = {
        "train": "train.npy",
        "validation": "validation.npy",
        "test": "test.npy",
    }

    for split, fname in split_map.items():
        if dm.dataset is None or split not in dm.dataset:
            continue

        texts = _extract_texts_for_split(dm, split, sep=sep, dataset_meta=meta)

        # optional: prompt-wrapping
        texts = _maybe_build_prompts(
            texts, dataset_name=args.dataset, use_prompts=(not args.no_prompts)
        )

        out_path = out_dir / fname
        N, D = _write_embeddings_npy(
            encoder=encoder,
            texts=texts,
            out_path=out_path,
            batch_size=args.batch_size,
            desc=f"{args.dataset}:{split}",
        )
        print(f"[{split}] wrote {out_path} with shape=({N}, {D})")

    # small metadata file
    (out_dir / "meta.txt").write_text(
        f"dataset={args.dataset}\nllm_name={args.llm_name}\nmax_seq_length={args.max_seq_length}\n"
        f"batch_size={args.batch_size}\nnormalize_embeddings={args.normalize_embeddings}\n"
        f"use_prompts={not args.no_prompts}\nsep={repr(sep)}\n"
    )


if __name__ == "__main__":
    main()
