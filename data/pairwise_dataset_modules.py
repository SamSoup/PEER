# data/pairwise_dataset_modules.py

from __future__ import annotations

from typing import Dict, Optional

from .base import PairSentenceRegressionDataModule

# Canonical dataset definitions for sentence-pair regression tasks.
# Keys must be lower-case canonical names.
PAIRWISE_DATASETS: Dict[str, Dict] = {
    "stsb": {
        "dataset_name": "sentence-transformers/stsb",
        "label_max": 1.0,
    },
    "stsbenchmark-mteb": {
        "dataset_name": "mteb/stsbenchmark-sts",
        "label_max": 5.0,
    },
    "sickr-sts": {
        "dataset_name": "samsoup/sickr-sts",
        "label_max": 5.0,
    },
    "sts22-crosslingual-sts": {
        "dataset_name": "Samsoup/sts22-crosslingual-sts",
        "label_max": 4.0,
    },
    "wmt20-en-zh": {
        "dataset_name": "samsoup/Samsoup-WMT2020-en-zh",
        "label_max": 100.0,
    },
    "wmt20-ru-en": {
        "dataset_name": "samsoup/Samsoup-WMT2020-ru-en",
        "label_max": 100.0,
    },
    "wmt20-si-en": {
        "dataset_name": "samsoup/Samsoup-WMT2020-si-en",
        "label_max": 100.0,
    },
}

# Normalizes commonly used aliases to canonical keys above.
PAIRWISE_NAME_ALIASES: Dict[str, str] = {
    "stsbenchmark": "stsbenchmark-mteb",
    "stsbench-mteb": "stsbenchmark-mteb",
    "sickr": "sickr-sts",
    "sickr_sts": "sickr-sts",
    "sickr-sts": "sickr-sts",
    "sts22-xling-sts": "sts22-crosslingual-sts",
    "sts22": "sts22-crosslingual-sts",
    "sts22_crosslingual_sts": "sts22-crosslingual-sts",
    "wmt20-enzh": "wmt20-en-zh",
    "wmt20-zhen": "wmt20-en-zh",
    "wmt20-zh-en": "wmt20-en-zh",
    "wmt_en_zh": "wmt20-en-zh",
    "wmt-en-zh": "wmt20-en-zh",
    "wmt20-ruen": "wmt20-ru-en",
    "wmt20-enru": "wmt20-ru-en",
    "wmt20-en-ru": "wmt20-ru-en",
    "wmt_en_ru": "wmt20-ru-en",
    "wmt-en-ru": "wmt20-ru-en",
    "wmt20-sien": "wmt20-si-en",
    "wmt20-ensi": "wmt20-si-en",
    "wmt20-en-en": "wmt20-si-en",
    "wmt_si_en": "wmt20-si-en",
    "wmt-si-en": "wmt20-si-en",
    "stsb": "stsb",
}


def _canonicalize_name(dataset_name: str) -> str:
    normalized = dataset_name.lower()
    return PAIRWISE_NAME_ALIASES.get(normalized, normalized)


def build_pairwise_datamodule(
    *,
    dataset_name: str,
    model_name: str,
    max_seq_length: int,
    batch_size: int,
    tokenize_inputs: bool = True,
    combine_fields: bool = False,
    combine_separator_token: str = "[SEP]",
) -> Optional[PairSentenceRegressionDataModule]:
    """
    Create a PairSentenceRegressionDataModule using registry metadata.
    Returns None if the dataset name is unsupported.
    """
    canonical = _canonicalize_name(dataset_name)
    config = PAIRWISE_DATASETS.get(canonical)
    if config is None:
        return None

    return PairSentenceRegressionDataModule(
        model_name_or_path=model_name,
        dataset_name=config["dataset_name"],
        dataset_config_name=config.get("dataset_config_name"),
        load_dataset_kwargs=config.get("load_dataset_kwargs"),
        sentence1_field=config.get("sentence1_field", "sentence1"),
        sentence2_field=config.get("sentence2_field", "sentence2"),
        output_column=config.get("output_column", "score"),
        label_max=config.get("label_max"),
        combine_fields=combine_fields,
        combine_separator_token=combine_separator_token,
        max_seq_length=max_seq_length,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        tokenize_inputs=tokenize_inputs,
    )


__all__ = [
    "PAIRWISE_DATASETS",
    "PAIRWISE_NAME_ALIASES",
    "build_pairwise_datamodule",
]
