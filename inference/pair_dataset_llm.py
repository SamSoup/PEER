import argparse
import json
import os
import random
from typing import List, Optional
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.pair_dataset_registry import (
    DATASET_ALL_KEYS,
    PROMPT_TEMPLATES,
    get_dataset_meta,
    parse_score,
)
from pair_dataset_api import (
    sample_icl_examples,
    compute_metrics,
    write_predictions,
    print_progress,
)


def ensure_hf_cache():
    """Ensure HF downloads go to a writable cache."""
    default_cache = "/scratch/06782/ysu707/.cache"
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = default_cache
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        os.path.join(os.environ["HF_HOME"], "transformers"),
    )
    os.environ.setdefault(
        "HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets")
    )
    os.environ.setdefault(
        "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub")
    )


def is_qwen3_model(model_id: str) -> bool:
    if not model_id:
        return False
    name = model_id.lower()
    return "qwen3" in name or "qwen-3" in name or "qwen 3" in name


def build_prompt(
    tokenizer, system_prompt: str, user_prompt: str, model_id: str
) -> str:
    """Prefer chat template; fall back to plain text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        extra_kwargs = (
            {"enable_thinking": False} if is_qwen3_model(model_id) else {}
        )
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
        except Exception:
            pass
    return system_prompt + "\n\n" + user_prompt + "\n"


def _resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Use one of: {list(mapping.keys()) + ['auto']}"
        )
    return mapping[dtype_str]


def load_local_model(model_path: str, device_map: str, torch_dtype: str):
    dtype = _resolve_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
        cache_dir=os.environ["HF_HOME"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def generate_local_completion(
    prompt_texts: List[str],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=20000,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output = model.generate(
            **inputs, **{k: v for k, v in gen_kwargs.items() if v is not None}
        )
    input_lengths = inputs["attention_mask"].sum(dim=1)
    completions: List[str] = []
    for i in range(output.shape[0]):
        completion_ids = output[i, input_lengths[i] :]
        completions.append(
            tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        )
    return completions


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (e.g., Qwen reasoning) before parsing."""
    if not text:
        return text
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def run_eval(
    dataset_key: str,
    model_path: str,
    out_dir: str,
    limit: Optional[int],
    icl_n: int,
    seed: int,
    use_cot: bool,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    progress_every: int,
) -> None:
    meta_cfg = get_dataset_meta(dataset_key)
    if meta_cfg is None:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    ensure_hf_cache()
    prompt_template = PROMPT_TEMPLATES[meta_cfg.prompt_key]

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # Lazy import to keep this file free of requests/tqdm deps except where used
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    ds = load_dataset(meta_cfg.hf_id)
    train_split = (
        ds["train"] if meta_cfg.uses_train_split and "train" in ds else None
    )
    test_split = ds["test"]

    if train_split is None and icl_n and meta_cfg.disable_icl_if_missing_train:
        print("No train split available; disabling ICL sampling.")
        icl_n = 0

    icl_examples = sample_icl_examples(train_split, icl_n, meta_cfg)

    model, tokenizer = load_local_model(model_path, device_map, torch_dtype)

    gold_buffer: List[float] = []
    pred_buffer: List[float] = []
    rows_for_dump = []

    total = len(test_split) if limit is None else min(len(test_split), limit)

    for start in tqdm(
        range(0, total, batch_size),
        total=(total + batch_size - 1) // batch_size,
    ):
        end = min(total, start + batch_size)
        batch_rows = [test_split[i] for i in range(start, end)]

        prompts: List[str] = []
        meta = []
        for idx_offset, row in enumerate(batch_rows):
            i = start + idx_offset
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            gold_raw = float(row[meta_cfg.score_field])

            user_msg = prompt_template.build_user_message(
                s1, s2, icl_examples, use_cot
            )
            prompt_text = build_prompt(
                tokenizer, prompt_template.system, user_msg, model_path
            )
            prompts.append(prompt_text)
            meta.append((i, row, gold_raw))

        try:
            raw_answers = generate_local_completion(
                prompt_texts=prompts,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"[batch {start}-{end}] ERROR generating with local model: {e}"
            )
            continue

        for (i, row, gold_raw), raw_answer in zip(meta, raw_answers):
            cleaned_answer = strip_think_tags(raw_answer)
            try:
                pred_raw = parse_score(cleaned_answer, meta_cfg)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(
                    f"[{i}] ERROR parsing model output: {e} | raw: {raw_answer}"
                )
                continue

            row_record = {
                "idx": i,
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "gold_score": gold_raw,
                "pred_score": pred_raw,
                "raw_model_reply": raw_answer,
            }

            gold_buffer.append(gold_raw)
            pred_buffer.append(pred_raw)

            rows_for_dump.append(row_record)
            print_progress(i, row_record, progress_every)

    metrics = compute_metrics(pred_buffer, gold_buffer)

    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    write_predictions(rows_for_dump, preds_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": meta_cfg.key,
                "model_name_or_path": model_path,
                "icl_n": int(len(icl_examples)),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    with open(icl_path, "w", encoding="utf-8") as f:
        json.dump(icl_examples, f, indent=2)

    print("=======================================")
    for name, val in metrics.items():
        if val is not None:
            print(f"{name.upper()}: {val:.3f}")
    print(f"In-context examples used: {len(icl_examples)}")
    print("---")

    print(f"Saved predictions to: {preds_path}")
    print(f"Saved metrics to:     {metrics_path}")
    print(f"Saved ICL examples to:{icl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified STS/translation evaluation using a local HF model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_ALL_KEYS,
        help="Which dataset to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model name or path for AutoModelForCausalLM.from_pretrained.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write predictions.csv, metrics.json, and icl_examples.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only score this many test pairs (for quick runs).",
    )
    parser.add_argument(
        "--icl_n",
        type=int,
        default=0,
        help="How many random training examples to include as in-context demonstrations. 0 disables ICL.",
    )
    parser.add_argument(
        "--use_cot",
        action="store_true",
        help='Append "Let\'s think step by step" to enable zero-shot chain-of-thought prompting while still returning a final numeric score.',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting in-context examples.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for model loading (e.g., "auto", "cuda:0").',
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help='Torch dtype for model weights: "auto", "fp32", "fp16", "bf16", or "int8".',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate for each completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 uses greedy decoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for local generation.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N evaluated pairs; set to 0 or below to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        dataset_key=args.dataset,
        model_path=args.model_path,
        out_dir=args.out_dir,
        limit=args.limit,
        icl_n=args.icl_n,
        seed=args.seed,
        use_cot=args.use_cot,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
