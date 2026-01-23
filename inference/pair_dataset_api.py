import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from datasets import load_dataset
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm.auto import tqdm

from prompts import PROMPT_TEMPLATES, PromptTemplate

DEFAULT_API_BASE_URL = "https://ai.tejas.tacc.utexas.edu"


@dataclass
class MetricScale:
    name: str
    gold_column: str
    pred_column: str
    gold_transform: Callable[[float], float]
    pred_transform: Callable[[float], float]
    label_desc: str


@dataclass
class DatasetConfig:
    key: str
    hf_id: str
    prompt_key: str
    score_range: Tuple[float, float]
    parse_pattern: str
    icl_formatter: Callable[[float], str]
    metric_scales: List[MetricScale]
    round_predictions: Optional[int] = 3
    uses_train_split: bool = True
    disable_icl_if_missing_train: bool = False
    description: str = ""
    max_retries: int = 3
    score_field: str = "score"


def clamp_and_round(
    value: float, bounds: Tuple[float, float], decimals: Optional[int]
) -> float:
    value = max(bounds[0], min(bounds[1], value))
    if decimals is None:
        return value
    return float(f"{value:.{decimals}f}")


def parse_score(raw_answer: str, config: DatasetConfig) -> float:
    matches = re.findall(config.parse_pattern, raw_answer.strip())
    if not matches:
        raise ValueError(
            f"Could not parse numeric similarity from: {raw_answer}"
        )
    last = matches[-1]
    if isinstance(last, tuple):
        last = last[0]
    score = float(last)
    return clamp_and_round(score, config.score_range, config.round_predictions)


def extract_answer_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, TypeError):
        return data["choices"][0].get("text")


def query_similarity_llm(
    api_base_url: str,
    api_key: str,
    model_name: str,
    prompt: PromptTemplate,
    sentence_a: str,
    sentence_b: str,
    icl_examples: List[dict],
    use_cot: bool,
    config: DatasetConfig,
    max_new_tokens: int,
    temperature: float,
    timeout: int = 60,
    backoff_seconds: int = 60,
) -> Tuple[float, str, dict]:
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
    }
    endpoint = api_base_url.rstrip("/") + "/chat/completions"

    last_error = None
    for attempt in range(1, config.max_retries + 1):
        try:
            resp = requests.post(
                endpoint, headers=headers, json=payload, timeout=timeout
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            last_error = e
            if attempt == config.max_retries:
                raise RuntimeError(
                    f"LLM request failed after {config.max_retries} attempts: {e}"
                )
            print(
                f"Request error (attempt {attempt}/{config.max_retries}): {e}. "
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
            parsed = parse_score(raw_answer, config)
            return parsed, raw_answer.strip(), data

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt == config.max_retries:
                raise RuntimeError(
                    f"LLM request failed after {config.max_retries} attempts: "
                    f"{resp.status_code} {resp.text}"
                )
            print(
                f"LLM request failed with status {resp.status_code} "
                f"(attempt {attempt}/{config.max_retries}). "
                f"Sleeping {backoff_seconds} seconds before retry."
            )
            time.sleep(backoff_seconds)
            continue

        raise RuntimeError(
            f"LLM request failed: {resp.status_code} {resp.text}"
        )

    raise RuntimeError(
        f"LLM request failed after {config.max_retries} attempts: {last_error}"
    )


def sample_icl_examples(
    train_split, n: int, config: DatasetConfig
) -> List[dict]:
    if n is None or n <= 0 or train_split is None or len(train_split) == 0:
        return []
    idxs = random.sample(range(len(train_split)), k=min(n, len(train_split)))
    examples = []
    for idx in idxs:
        row = train_split[idx]
        score_raw = float(row[config.score_field])
        examples.append(
            {
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score_text": config.icl_formatter(score_raw),
            }
        )
    return examples


def mse(pred: np.ndarray, gold: np.ndarray) -> float:
    return float(np.mean((pred - gold) ** 2))


def safe_corr(
    fn: Callable, pred: np.ndarray, gold: np.ndarray
) -> Optional[float]:
    if len(pred) < 2:
        return None
    try:
        val, _ = fn(pred, gold)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if val is None or (
        isinstance(val, float) and (math.isnan(val) or math.isinf(val))
    ):
        return None
    return float(val)


def format_for_csv(val: float) -> str:
    return f"{val:.3f}"


def format_metric(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return float(f"{val:.3f}")


def compute_metrics(
    buffers: Dict[str, Dict[str, List[float]]], scales: List[MetricScale]
) -> Dict[str, dict]:
    metrics: Dict[str, dict] = {}
    for scale in scales:
        gold = np.asarray(buffers[scale.name]["gold"], dtype=float)
        pred = np.asarray(buffers[scale.name]["pred"], dtype=float)

        mse_val = mse(pred, gold) if len(pred) else None
        rmse_val = float(np.sqrt(mse_val)) if mse_val is not None else None

        scale_metrics = {
            "mse": format_metric(mse_val),
            "rmse": format_metric(rmse_val),
            "pearson_correlation": format_metric(safe_corr(pearsonr, pred, gold)),
            "spearman_correlation": format_metric(safe_corr(spearmanr, pred, gold)),
            "kendall_correlation": format_metric(safe_corr(kendalltau, pred, gold)),
            "label_range": scale.label_desc,
            "num_pairs_scored": int(len(pred)),
        }
        metrics[scale.name] = scale_metrics
    return metrics


def esc(val: str) -> str:
    return '"' + str(val).replace('"', '""') + '"'


def write_predictions(
    rows: List[dict], path: str, scales: List[MetricScale]
) -> None:
    if not rows:
        return
    columns = ["idx", "sentence1", "sentence2"]
    for scale in scales:
        columns.extend([scale.gold_column, scale.pred_column])
    columns.append("raw_model_reply")

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            cells = [
                str(row["idx"]),
                esc(row["sentence1"]),
                esc(row["sentence2"]),
            ]
            for scale in scales:
                cells.append(format_for_csv(row[scale.gold_column]))
                cells.append(format_for_csv(row[scale.pred_column]))
            cells.append(esc(row["raw_model_reply"]))
            f.write(",".join(cells) + "\n")


def print_progress(
    i: int, row: dict, scales: List[MetricScale], progress_every: int
) -> None:
    if progress_every is None or progress_every <= 0:
        return
    if i % progress_every != 0:
        return
    print(f"[{i}]")
    print("Sentence 1:", row["sentence1"])
    print("Sentence 2:", row["sentence2"])
    first_scale = scales[0]
    print(
        f"Gold ({first_scale.gold_column}): {row[first_scale.gold_column]:.3f}"
    )
    print(
        f"LLM  ({first_scale.pred_column}): {row[first_scale.pred_column]:.3f}"
    )
    print("Raw model reply:", row["raw_model_reply"])
    print()


def build_dataset_configs() -> Dict[str, DatasetConfig]:
    return {
        "stsb": DatasetConfig(
            key="stsb",
            hf_id="sentence-transformers/stsb",
            prompt_key="stsb",
            score_range=(0.0, 1.0),
            parse_pattern=r"\b([01](?:\.\d+)?|\d?\.\d+)\b",
            icl_formatter=lambda raw: f"{raw / 5.0:.3f}",
            round_predictions=None,
            description="STS-B evaluated on 0-1 normalized scores.",
            metric_scales=[
                MetricScale(
                    name="normalized_0_1",
                    gold_column="gold_similarity_0_1",
                    pred_column="llm_similarity_0_1",
                    gold_transform=lambda raw: float(raw) / 5.0,
                    pred_transform=lambda pred: pred,
                    label_desc="Normalized scores within 0..1 (raw score divided by 5).",
                )
            ],
        ),
        "stsbenchmark_mteb": DatasetConfig(
            key="stsbenchmark_mteb",
            hf_id="mteb/stsbenchmark-sts",
            prompt_key="stsbenchmark_mteb",
            score_range=(0.0, 5.0),
            parse_pattern=r"\b\d+(?:\.\d+)?\b",
            icl_formatter=lambda raw: f"{float(raw):.3f}",
            description="MTEB STS Benchmark scored on 0-5 with normalized metrics.",
            metric_scales=[
                MetricScale(
                    name="raw",
                    gold_column="gold_similarity_raw",
                    pred_column="llm_similarity_raw",
                    gold_transform=lambda raw: float(raw),
                    pred_transform=lambda pred: pred,
                    label_desc="Raw scores between 0 and 5.",
                ),
                MetricScale(
                    name="normalized",
                    gold_column="gold_similarity_normalized",
                    pred_column="llm_similarity_normalized",
                    gold_transform=lambda raw: float(raw) / 5.0,
                    pred_transform=lambda pred: pred / 5.0,
                    label_desc="Normalized scores within 0..1 after dividing raw by 5.",
                ),
            ],
        ),
        "sickr_sts": DatasetConfig(
            key="sickr_sts",
            hf_id="Samsoup/sickr-sts",
            prompt_key="sickr_sts",
            score_range=(1.0, 5.0),
            parse_pattern=r"\b\d+(?:\.\d+)?\b",
            icl_formatter=lambda raw: f"{float(raw):.3f}",
            description="SICK-R STS scored on 1-5 with optional normalization.",
            disable_icl_if_missing_train=True,
            max_retries=2,
            metric_scales=[
                MetricScale(
                    name="raw",
                    gold_column="gold_similarity_raw",
                    pred_column="llm_similarity_raw",
                    gold_transform=lambda raw: float(raw),
                    pred_transform=lambda pred: pred,
                    label_desc="Raw scores between 1 and 5.",
                ),
                MetricScale(
                    name="normalized",
                    gold_column="gold_similarity_normalized",
                    pred_column="llm_similarity_normalized",
                    gold_transform=lambda raw: (float(raw) - 1.0) / 4.0,
                    pred_transform=lambda pred: (pred - 1.0) / 4.0,
                    label_desc="Normalized scores within 0..1 after (raw-1)/4.",
                ),
            ],
        ),
        "sts22": DatasetConfig(
            key="sts22",
            hf_id="Samsoup/sts22-crosslingual-sts",
            prompt_key="sts22",
            score_range=(1.0, 4.0),
            parse_pattern=r"\b\d+(?:\.\d+)?\b",
            icl_formatter=lambda raw: f"{float(raw):.3f}",
            description="STS22 cross-lingual scored on 1-4 with normalized metrics.",
            disable_icl_if_missing_train=True,
            max_retries=2,
            metric_scales=[
                MetricScale(
                    name="raw",
                    gold_column="gold_similarity_raw",
                    pred_column="llm_similarity_raw",
                    gold_transform=lambda raw: float(raw),
                    pred_transform=lambda pred: pred,
                    label_desc="Raw scores between 1 and 4.",
                ),
                MetricScale(
                    name="normalized",
                    gold_column="gold_similarity_normalized",
                    pred_column="llm_similarity_normalized",
                    gold_transform=lambda raw: (float(raw) - 1.0) / 3.0,
                    pred_transform=lambda pred: (pred - 1.0) / 3.0,
                    label_desc="Normalized scores within 0..1 after (raw-1)/3.",
                ),
            ],
        ),
        "wmt_en_ru": DatasetConfig(
            key="wmt_en_ru",
            hf_id="samsoup/Samsoup-WMT2020-ru-en",
            prompt_key="wmt_en_ru",
            score_range=(0.0, 100.0),
            parse_pattern=r"(-?\d+(?:\.\d+)?)",
            icl_formatter=lambda raw: f"{float(raw):.3f}",
            description="WMT20 Russian→English scored on 0-100 with normalized metrics.",
            metric_scales=[
                MetricScale(
                    name="raw",
                    gold_column="gold_score_raw",
                    pred_column="llm_score_raw",
                    gold_transform=lambda raw: float(raw),
                    pred_transform=lambda pred: pred,
                    label_desc="Raw scores between 0 and 100.",
                ),
                MetricScale(
                    name="normalized",
                    gold_column="gold_score_normalized",
                    pred_column="llm_score_normalized",
                    gold_transform=lambda raw: float(raw) / 100.0,
                    pred_transform=lambda pred: pred / 100.0,
                    label_desc="Normalized scores within 0..1 after dividing raw by 100.",
                ),
            ],
        ),
        "wmt_en_zh": DatasetConfig(
            key="wmt_en_zh",
            hf_id="samsoup/Samsoup-WMT2020-en-zh",
            prompt_key="wmt_en_zh",
            score_range=(0.0, 100.0),
            parse_pattern=r"(-?\d+(?:\.\d+)?)",
            icl_formatter=lambda raw: f"{float(raw):.3f}",
            description="WMT20 English→Chinese scored on 0-100 with normalized metrics.",
            metric_scales=[
                MetricScale(
                    name="raw",
                    gold_column="gold_score_raw",
                    pred_column="llm_score_raw",
                    gold_transform=lambda raw: float(raw),
                    pred_transform=lambda pred: pred,
                    label_desc="Raw scores between 0 and 100.",
                ),
                MetricScale(
                    name="normalized",
                    gold_column="gold_score_normalized",
                    pred_column="llm_score_normalized",
                    gold_transform=lambda raw: float(raw) / 100.0,
                    pred_transform=lambda pred: pred / 100.0,
                    label_desc="Normalized scores within 0..1 after dividing raw by 100.",
                ),
            ],
        ),
        "wmt_si_en": DatasetConfig(
            key="wmt_si_en",
            hf_id="samsoup/Samsoup-WMT2020-si-en",
            prompt_key="wmt_si_en",
            score_range=(0.0, 100.0),
            parse_pattern=r"(-?\d+(?:\.\d+)?)",
            icl_formatter=lambda raw: f"{float(raw):.3f}",
            description="WMT20 Sinhala→English scored on 0-100 with normalized metrics.",
            metric_scales=[
                MetricScale(
                    name="raw",
                    gold_column="gold_score_raw",
                    pred_column="llm_score_raw",
                    gold_transform=lambda raw: float(raw),
                    pred_transform=lambda pred: pred,
                    label_desc="Raw scores between 0 and 100.",
                ),
                MetricScale(
                    name="normalized",
                    gold_column="gold_score_normalized",
                    pred_column="llm_score_normalized",
                    gold_transform=lambda raw: float(raw) / 100.0,
                    pred_transform=lambda pred: pred / 100.0,
                    label_desc="Normalized scores within 0..1 after dividing raw by 100.",
                ),
            ],
        ),
    }


DATASET_CONFIGS = build_dataset_configs()


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
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    config = DATASET_CONFIGS[dataset_key]
    prompt_template = PROMPT_TEMPLATES[config.prompt_key]

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    ds = load_dataset(config.hf_id)
    train_split = (
        ds["train"] if config.uses_train_split and "train" in ds else None
    )
    test_split = ds["test"]

    if train_split is None and icl_n and config.disable_icl_if_missing_train:
        print("No train split available; disabling ICL sampling.")
        icl_n = 0

    icl_examples = sample_icl_examples(train_split, icl_n, config)

    buffers: Dict[str, Dict[str, List[float]]] = {
        scale.name: {"gold": [], "pred": []} for scale in config.metric_scales
    }
    rows_for_dump: List[dict] = []

    for i, row in tqdm(enumerate(test_split), total=len(test_split)):
        if limit is not None and i >= limit:
            break

        s1 = row["sentence1"]
        s2 = row["sentence2"]
        gold_raw = float(row[config.score_field])

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
                config=config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[{i}] ERROR querying model: {e}")
            continue

        row_record = {
            "idx": i,
            "sentence1": s1,
            "sentence2": s2,
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
        print_progress(i, row_record, config.metric_scales, progress_every)

    metrics = compute_metrics(buffers, config.metric_scales)

    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    write_predictions(rows_for_dump, preds_path, config.metric_scales)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_key,
                "model_name": model_name,
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
            print(f"Pearson: {scale_metrics['pearson_correlation']:.3f}")
        if scale_metrics["spearman_correlation"] is not None:
            print(f"Spearman: {scale_metrics['spearman_correlation']:.3f}")
        if scale_metrics["kendall_correlation"] is not None:
            print(f"Kendall τ: {scale_metrics['kendall_correlation']:.3f}")
        if scale_metrics["mse"] is not None:
            print(f"MSE:  {scale_metrics['mse']:.3f}")
            print(f"RMSE: {scale_metrics['rmse']:.3f}")
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
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Which dataset to evaluate (keys: {}).".format(
            ", ".join(sorted(DATASET_CONFIGS.keys()))
        ),
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=DEFAULT_API_BASE_URL,
        help="API base URL (without /chat/completions). Defaults to tejas.tacc endpoint.",
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
