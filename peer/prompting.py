"""
Prompt helpers that mirror the dataset->prompt mapping in inference/prompts.py.
Builds a full prompt string (system + user message) for a given dataset key.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Iterable, List, Tuple, Union


def _load_prompt_templates():
    """
    Load PROMPT_TEMPLATES from inference/prompts.py without requiring a package import.
    """
    base_dir = Path(__file__).resolve().parent.parent
    prompt_path = base_dir / "inference" / "prompts.py"
    spec = importlib.util.spec_from_file_location(
        "modelsv2_prompt_templates", prompt_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module.PROMPT_TEMPLATES


PROMPT_TEMPLATES = _load_prompt_templates()

RawText = Union[str, Tuple[str, str], List[str], List[Tuple[str, str]]]


# Map dataset names/aliases -> prompt template keys in inference/prompts.py
DATASET_PROMPT_MAP = {
    "stsb": "stsb",
    "stsbenchmark": "stsbenchmark_mteb",
    "stsbenchmark-mteb": "stsbenchmark_mteb",
    "stsbenchmark_mteb": "stsbenchmark_mteb",
    "stsbench-mteb": "stsbenchmark_mteb",
    "sickr": "sickr_sts",
    "sickr-sts": "sickr_sts",
    "sickr_sts": "sickr_sts",
    "sts22": "sts22",
    "sts22-crosslingual-sts": "sts22",
    "sts22_crosslingual_sts": "sts22",
    "sts22-xling-sts": "sts22",
    "wmt20-ru-en": "wmt_en_ru",
    "wmt20_ru_en": "wmt_en_ru",
    "wmt20-en-ru": "wmt_en_ru",
    "wmt_en_ru": "wmt_en_ru",
    "wmt20-en-zh": "wmt_en_zh",
    "wmt20_en_zh": "wmt_en_zh",
    "wmt_en_zh": "wmt_en_zh",
    "wmt20-si-en": "wmt_si_en",
    "wmt20_si_en": "wmt_si_en",
    "wmt_si_en": "wmt_si_en",
}

DEFAULT_TMPL = "Input:\n{text}\n\nPredict a score from 1 to 4:"


def canonical_dataset_name(name: str | None) -> str | None:
    if not name:
        return None
    return name.strip().lower().replace(" ", "")


def prompt_key_for_dataset(dataset_name: str | None) -> str | None:
    canonical = canonical_dataset_name(dataset_name)
    if canonical is None:
        return None
    return DATASET_PROMPT_MAP.get(canonical, canonical if canonical in PROMPT_TEMPLATES else None)


def _maybe_split_combined(text: str) -> Tuple[str, str] | None:
    """
    If a combined string contains a clear separator, split into a pair.
    """
    if " [SEP] " in text:
        a, b = text.split(" [SEP] ", 1)
        return a, b
    if "||" in text:
        a, b = text.split("||", 1)
        return a.strip(), b.strip()
    return None


def build_prompt_for_example(text: RawText, dataset_name: str | None = None) -> str:
    """
    Build a single prompt string for a raw example (string or (s1,s2)).
    Falls back to a generic template when no dataset-specific prompt exists.
    """
    key = prompt_key_for_dataset(dataset_name)
    tmpl = PROMPT_TEMPLATES.get(key) if key else None

    # Normalize to pair when possible
    if isinstance(text, (list, tuple)) and len(text) == 2:
        pair = (str(text[0]), str(text[1]))
    elif isinstance(text, str):
        split = _maybe_split_combined(text)
        pair = split if split else None
    else:
        pair = None

    if tmpl is not None and pair is not None:
        user_msg = tmpl.build_user_message(
            pair[0], pair[1], icl_examples=[], use_cot=False
        )
        return f"{tmpl.system}\n\n{user_msg}"

    # Fallback to a generic single-string prompt
    if isinstance(text, str):
        raw = text
    elif isinstance(text, (list, tuple)):
        raw = " [SEP] ".join([str(t) for t in text])
    else:
        raw = str(text)
    return DEFAULT_TMPL.format(text=raw)


def build_prompts(texts: Iterable[RawText], dataset_name: str | None = None) -> List[str]:
    """Vectorized helper over a batch of raw texts."""
    return [build_prompt_for_example(t, dataset_name) for t in texts]
