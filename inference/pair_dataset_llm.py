import argparse
import json
import os
import random
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPT_TEMPLATES
from pair_dataset_api import (
    DATASET_CONFIGS,
    parse_score,
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


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Prefer chat template; fall back to plain text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
        torch_dtype=dtype,
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
) -> None:
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    ensure_hf_cache()
    config = DATASET_CONFIGS[dataset_key]
    prompt_template = PROMPT_TEMPLATES[config.prompt_key]

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # Lazy import to keep this file free of requests/tqdm deps except where used
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    ds = load_dataset(config.hf_id)
    train_split = (
        ds["train"] if config.uses_train_split and "train" in ds else None
    )
    test_split = ds["test"]

    if train_split is None and icl_n and config.disable_icl_if_missing_train:
        print("No train split available; disabling ICL sampling.")
        icl_n = 0

    icl_examples = sample_icl_examples(train_split, icl_n, config)

    model, tokenizer = load_local_model(model_path, device_map, torch_dtype)

    buffers = {
        scale.name: {"gold": [], "pred": []} for scale in config.metric_scales
    }
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
            gold_raw = float(row[config.score_field])

            user_msg = prompt_template.build_user_message(
                s1, s2, icl_examples, use_cot
            )
            prompt_text = build_prompt(
                tokenizer, prompt_template.system, user_msg
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
            try:
                pred_raw = parse_score(raw_answer, config)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(
                    f"[{i}] ERROR parsing model output: {e} | raw: {raw_answer}"
                )
                continue

            row_record = {
                "idx": i,
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "raw_model_reply": raw_answer,
            }

            for scale in config.metric_scales:
                gold_val = scale.gold_transform(gold_raw)
                pred_val = scale.pred_transform(pred_raw)
                buffers[scale.name]["gold"].append(gold_val)
                buffers[scale.name]["pred"].append(pred_val)
                row_record[scale.gold_column] = gold_val
                row_record[scale.pred_column] = pred_val

            rows_for_dump.append(row_record)
            print_progress(i, row_record, config.metric_scales)

    metrics = compute_metrics(buffers, config.metric_scales)

    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    write_predictions(rows_for_dump, preds_path, config.metric_scales)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_key,
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
    for scale in config.metric_scales:
        scale_metrics = metrics[scale.name]
        print(f"{scale.name.upper()} ({scale.label_desc})")
        if scale_metrics["pearson_correlation"] is not None:
            print(f"Pearson: {scale_metrics['pearson_correlation']:.4f}")
        if scale_metrics["spearman_correlation"] is not None:
            print(f"Spearman: {scale_metrics['spearman_correlation']:.4f}")
        if scale_metrics["kendall_correlation"] is not None:
            print(f"Kendall τ: {scale_metrics['kendall_correlation']:.4f}")
        if scale_metrics["mse"] is not None:
            print(f"MSE:  {scale_metrics['mse']:.6f}")
            print(f"RMSE: {scale_metrics['rmse']:.6f}")
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
        choices=sorted(DATASET_CONFIGS.keys()),
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
    )
