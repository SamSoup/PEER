import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import requests
from datasets import load_dataset
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm.auto import tqdm

from data.pair_dataset_registry import (
    DATASET_ALL_KEYS,
    PROMPT_TEMPLATES,
    DatasetMeta,
    format_score,
    get_dataset_meta,
    parse_score,
)


def extract_answer_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, TypeError):
        return data["choices"][0].get("text")


def query_similarity_llm(
    api_base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    sentence_a: str,
    sentence_b: str,
    icl_examples: List[dict],
    use_cot: bool,
    meta: DatasetMeta,
    max_new_tokens: int,
    temperature: float,
    timeout: int = 60,
    backoff_seconds: int = 60,
    seed: int = 42,
) -> tuple[float, str, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    user_msg = prompt.build_user_message(
        sentence_a, sentence_b, icl_examples, use_cot
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    endpoint = api_base_url.rstrip("/") + "/chat/completions"

    last_error: Optional[Exception] = None
    for attempt in range(1, meta.max_retries + 1):
        try:
            resp = requests.post(
                endpoint, headers=headers, json=payload, timeout=timeout
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            last_error = e
            if attempt == meta.max_retries:
                raise RuntimeError(
                    f"LLM request failed after {meta.max_retries} attempts: {e}"
                )
            print(
                f"Request error (attempt {attempt}/{meta.max_retries}): {e}. "
                f"Sleeping {backoff_seconds} seconds before retry."
            )
            time.sleep(backoff_seconds)
            continue

        if resp.status_code == 200:
            data = resp.json()
            raw_answer = extract_answer_text(data)
            if raw_answer is None:
                raise RuntimeError(
                    f"Could not find completion text in response:\n{data}"
                )
            parsed = parse_score(raw_answer, meta)
            return parsed, raw_answer.strip(), data

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt == meta.max_retries:
                raise RuntimeError(
                    f"LLM request failed after {meta.max_retries} attempts: "
                    f"{resp.status_code} {resp.text}"
                )
            print(
                f"LLM request failed with status {resp.status_code} "
                f"(attempt {attempt}/{meta.max_retries}). "
                f"Sleeping {backoff_seconds} seconds before retry."
            )
            time.sleep(backoff_seconds)
            continue

        raise RuntimeError(
            f"LLM request failed: {resp.status_code} {resp.text}"
        )

    raise RuntimeError(
        f"LLM request failed after {meta.max_retries} attempts: {last_error}"
    )


def sample_icl_examples(train_split, n: int, meta: DatasetMeta) -> List[dict]:
    if n is None or n <= 0 or train_split is None or len(train_split) == 0:
        return []
    idxs = random.sample(range(len(train_split)), k=min(n, len(train_split)))
    examples = []
    for idx in idxs:
        row = train_split[idx]
        score_raw = float(row[meta.score_field])
        examples.append(
            {
                "sentence1": row[meta.sentence1_field],
                "sentence2": row[meta.sentence2_field],
                "score_text": format_score(score_raw),
            }
        )
    return examples


def mse(pred: np.ndarray, gold: np.ndarray) -> float:
    return float(np.mean((pred - gold) ** 2))


def safe_corr(fn, pred: np.ndarray, gold: np.ndarray) -> Optional[float]:
    if len(pred) < 2:
        return None
    try:
        val, _ = fn(pred, gold)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if val is None or (
        isinstance(val, float) and (np.isnan(val) or np.isinf(val))
    ):
        return None
    return float(val)


def compute_metrics(
    pred: List[float], gold: List[float]
) -> Dict[str, Optional[float]]:
    if not pred or not gold:
        return {
            "mse": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "kendall": None,
        }
    pred_arr = np.asarray(pred, dtype=float)
    gold_arr = np.asarray(gold, dtype=float)
    mse_val = mse(pred_arr, gold_arr)
    return {
        "mse": mse_val,
        "rmse": float(np.sqrt(mse_val)),
        "pearson": safe_corr(pearsonr, pred_arr, gold_arr),
        "spearman": safe_corr(spearmanr, pred_arr, gold_arr),
        "kendall": safe_corr(kendalltau, pred_arr, gold_arr),
    }


def format_for_csv(val: float) -> str:
    return f"{val:.3f}"


def esc(val: str) -> str:
    return '"' + str(val).replace('"', '""') + '"'


def write_predictions(rows: List[dict], path: str) -> None:
    if not rows:
        return
    columns = [
        "idx",
        "sentence1",
        "sentence2",
        "gold_score",
        "pred_score",
        "raw_model_reply",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            cells = [
                str(row["idx"]),
                esc(row["sentence1"]),
                esc(row["sentence2"]),
                format_for_csv(row["gold_score"]),
                format_for_csv(row["pred_score"]),
                esc(row["raw_model_reply"]),
            ]
            f.write(",".join(cells) + "\n")


def print_progress(i: int, row: dict, progress_every: int = 100) -> None:
    if progress_every is None or progress_every <= 0:
        return
    if i % progress_every != 0:
        return
    print(f"[{i}]")
    print("Sentence 1:", row["sentence1"])
    print("Sentence 2:", row["sentence2"])
    print(f"Gold: {row['gold_score']:.3f}")
    print(f"LLM:  {row['pred_score']:.3f}")
    print("Raw model reply:", row["raw_model_reply"])
    print()


def run_eval(
    dataset_key: str,
    api_base_url: str,
    api_key: str,
    model_name: str,
    out_dir: str,
    limit: Optional[int],
    icl_n: int,
    seed: int,
    use_cot: bool,
    max_new_tokens: int,
    temperature: float,
    progress_every: int,
) -> None:
    meta = get_dataset_meta(dataset_key)
    if meta is None:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    prompt_template = PROMPT_TEMPLATES[meta.prompt_key]

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    ds = load_dataset(meta.hf_id)
    train_split = (
        ds["train"] if meta.uses_train_split and "train" in ds else None
    )
    test_split = ds["test"]
    if limit is not None:
        print(f"Truncating dataset to the first {limit} samples")
        test_split = test_split.select(range(limit))

    if train_split is None and icl_n and meta.disable_icl_if_missing_train:
        print("No train split available; disabling ICL sampling.")
        icl_n = 0

    icl_examples = sample_icl_examples(train_split, icl_n, meta)

    gold_buffer: List[float] = []
    pred_buffer: List[float] = []
    rows_for_dump: List[dict] = []

    for i, row in tqdm(enumerate(test_split), total=len(test_split)):

        s1 = row[meta.sentence1_field]
        s2 = row[meta.sentence2_field]
        gold_raw = float(row[meta.score_field])

        try:
            pred_raw, raw_answer, _ = query_similarity_llm(
                api_base_url=api_base_url,
                api_key=api_key,
                model_name=model_name,
                prompt=prompt_template,
                sentence_a=s1,
                sentence_b=s2,
                icl_examples=icl_examples,
                use_cot=use_cot,
                meta=meta,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[{i}] ERROR querying model: {e}")
            continue

        row_record = {
            "idx": i,
            "sentence1": s1,
            "sentence2": s2,
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
                "dataset": meta.key,
                "model_name": model_name,
                "icl_n": int(len(icl_examples)),
                "metrics": metrics,
                "seed": seed,
                "temperature": temperature,
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
        description="Unified STS/translation evaluation for multiple pair datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_ALL_KEYS,
        help="Which dataset to evaluate.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        help="API base URL (without /chat/completions).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Bearer token for the chat/completions endpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help='Model name to send in the "model" field.',
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
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to request from the model for each score.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model request.",
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
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        out_dir=args.out_dir,
        limit=args.limit,
        icl_n=args.icl_n,
        seed=args.seed,
        use_cot=args.use_cot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        progress_every=args.progress_every,
    )
