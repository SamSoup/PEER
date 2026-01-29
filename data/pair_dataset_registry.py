"""
Central registry for pairwise regression / STS-style datasets and prompt templates.

Goals:
- Single source of truth for:
    * dataset metadata (HF id, fields, score range)
    * prompt templates (system + formatting)
    * parsing/clamping helpers
    * alias resolution
- Clean global helper functions; no classes beyond dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union


# =========================
# Prompt templates
# =========================


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    example_header: str
    pair_intro: str
    score_instruction: str
    example_formatter: Callable[[dict, int], str]
    cot_suffix: str = (
        "Let's think step by step. After your reasoning, provide ONLY the final numeric score."
    )

    def build_user_message(
        self,
        sentence_a: str,
        sentence_b: str,
        icl_examples: List[dict],
        use_cot: bool,
    ) -> str:
        parts: List[str] = []
        if icl_examples:
            parts.append(self.example_header)
            for i, ex in enumerate(icl_examples, start=1):
                parts.append(self.example_formatter(ex, i))
            parts.append("\nNow score the new pair.\n")

        parts.append(
            self.pair_intro.format(
                sentence_a=sentence_a.strip(),
                sentence_b=sentence_b.strip(),
            )
        )
        parts.append(self.score_instruction)
        if use_cot:
            parts.append(self.cot_suffix)
        return "\n".join(parts)


def _fmt_ex(ex: dict, idx: int, header: str) -> str:
    return (
        f"Example {idx}:\n"
        f"Sentence A: {ex['sentence1']}\n"
        f"Sentence B: {ex['sentence2']}\n"
        f"{header}: {ex['score_text']}\n"
    )


def _fmt_ex_ru_en(ex: dict, idx: int) -> str:
    return (
        f"Example {idx}:\n"
        f"Sentence A (Russian source): {ex['sentence1']}\n"
        f"Sentence B (English translation): {ex['sentence2']}\n"
        f"Score (0-100): {ex['score_text']}\n"
    )


def _fmt_ex_en_zh(ex: dict, idx: int) -> str:
    return (
        f"Example {idx}:\n"
        f"Sentence A (English source): {ex['sentence1']}\n"
        f"Sentence B (Chinese translation): {ex['sentence2']}\n"
        f"Score (0-100): {ex['score_text']}\n"
    )


def _fmt_ex_si_en(ex: dict, idx: int) -> str:
    return (
        f"Example {idx}:\n"
        f"Sentence A (Sinhala source): {ex['sentence1']}\n"
        f"Sentence B (English translation): {ex['sentence2']}\n"
        f"Score (0-100): {ex['score_text']}\n"
    )


PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "stsb": PromptTemplate(
        system=(
            "You are an expert human annotator for semantic textual similarity (STS).\n"
            "You will see English sentence pairs and you must judge how similar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 0.000 to 1.000.\n"
            "- 1.000 = same meaning / paraphrases.\n"
            "- 0.500 = partially similar meaning but with important differences or missing info.\n"
            "- 0.000 = completely unrelated meaning.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity:\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction="Score (0.000 to 1.000 only):",
        example_formatter=lambda ex, idx: _fmt_ex(ex, idx, "Similarity (0-1)"),
    ),
    "stsbenchmark_mteb": PromptTemplate(
        system=(
            "You are an expert human annotator for semantic textual similarity (STS).\n"
            "You will see English sentence pairs and judge how similar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 0.000 to 5.000, inclusive.\n"
            "- 5.000 = same meaning / paraphrases.\n"
            "- 2.500 ≈ partially similar with important differences.\n"
            "- 0.000 = completely unrelated meaning.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score, with EXACTLY three digits after the decimal point (e.g., 3.000).\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity:\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction="Score (0.000 to 5.000 only):",
        example_formatter=lambda ex, idx: _fmt_ex(ex, idx, "Similarity (0-5)"),
    ),
    "sickr_sts": PromptTemplate(
        system=(
            "You are an expert annotator for semantic textual similarity (STS) for Sentences Involving "
            "Compositional Knowldedge.\n"
            "You will see sentence pairs and judge how similar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 1.000 to 5.000, inclusive.\n"
            "- 5.000 = same meaning / paraphrases; 1.000 = completely unrelated.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score, with EXACTLY three digits after the decimal point (e.g., 3.000).\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity:\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction="Score (1.000 to 5.000 only):",
        example_formatter=lambda ex, idx: _fmt_ex(ex, idx, "Similarity (1-5)"),
    ),
    "sts22": PromptTemplate(
        system=(
            "You are an expert annotator for semantic textual similarity (STS).\n"
            "You will see news article pairs that may be in DIFFERENT languages; judge how dissimilar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 1.000 to 4.000, inclusive.\n"
            "- 4.000 = NO similarity at all; 1.000 = two stories are almost identical.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score, with EXACTLY three digits after the decimal point (e.g., 3.000).\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity (pairs may be cross-lingual):\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction="Score (1.000 to 4.000 only):",
        example_formatter=lambda ex, idx: _fmt_ex(ex, idx, "Similarity (1-4)"),
    ),
    "wmt_en_ru": PromptTemplate(
        system=(
            "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
            "You will see sentence pairs where Sentence A is in Russian (source) and Sentence B is in English (translation).\n"
            "You must judge how well the meaning is preserved from Russian to English.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued translation quality / similarity score from 0.000 to 100.000.\n"
            "- 100.000 = perfect meaning preservation (excellent translation / paraphrases).\n"
            "- 50.000  = partially similar meaning but with important differences or missing info.\n"
            "- 0.000   = completely unrelated or very bad translation.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score Russian (source) to English (translation) quality / similarity:\n",
        pair_intro=(
            "Sentence A (Russian source): {sentence_a}\n"
            "Sentence B (English translation): {sentence_b}\n\n"
        ),
        score_instruction="Score (0.000 to 100.000 only):",
        example_formatter=_fmt_ex_ru_en,
    ),
    "wmt_en_zh": PromptTemplate(
        system=(
            "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
            "You will see sentence pairs where Sentence A is in English (source) and Sentence B is in Chinese (translation).\n"
            "You must judge how well the meaning is preserved from English to Chinese.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued translation quality / similarity score from 0.000 to 100.000.\n"
            "- 100.000 = perfect meaning preservation (excellent translation / paraphrases).\n"
            "- 50.000  = partially similar meaning but with important differences or missing info.\n"
            "- 0.000   = completely unrelated or very bad translation.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score English (source) to Chinese (translation) quality / similarity:\n",
        pair_intro=(
            "Sentence A (English source): {sentence_a}\n"
            "Sentence B (Chinese translation): {sentence_b}\n\n"
        ),
        score_instruction="Score (0.000 to 100.000 only):",
        example_formatter=_fmt_ex_en_zh,
    ),
    "wmt_si_en": PromptTemplate(
        system=(
            "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
            "You will see sentence pairs where Sentence A is in Sinhala (source) and Sentence B is in English (translation).\n"
            "You must judge how well the meaning is preserved from Sinhala to English.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued translation quality / similarity score from 0.000 to 100.000.\n"
            "- 100.000 = perfect meaning preservation (excellent translation / paraphrases).\n"
            "- 50.000  = partially similar meaning but with important differences or missing info.\n"
            "- 0.000   = completely unrelated or very bad translation.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score Sinhala (source) to English (translation) quality / similarity:\n",
        pair_intro=(
            "Sentence A (Sinhala source): {sentence_a}\n"
            "Sentence B (English translation): {sentence_b}\n\n"
        ),
        score_instruction="Score (0.000 to 100.000 only):",
        example_formatter=_fmt_ex_si_en,
    ),
}


# =========================
# Dataset metadata
# =========================


@dataclass(frozen=True)
class DatasetMeta:
    key: str
    hf_id: str
    prompt_key: str
    score_range: Tuple[float, float]
    parse_pattern: str
    description: str

    round_predictions: Optional[int] = 3
    max_retries: int = 3

    score_field: str = "score"
    sentence1_field: str = "sentence1"
    sentence2_field: str = "sentence2"

    dataset_config_name: Optional[str] = None
    load_dataset_kwargs: Optional[dict] = None


DATASET_METAS: Dict[str, DatasetMeta] = {
    "stsb": DatasetMeta(
        key="stsb",
        hf_id="sentence-transformers/stsb",
        prompt_key="stsb",
        score_range=(0.0, 1.0),
        parse_pattern=r"\b([01](?:\.\d+)?|\d?\.\d+)\b",
        description="STS-B evaluated on 0-1 raw scores.",
    ),
    "stsbenchmark_mteb": DatasetMeta(
        key="stsbenchmark_mteb",
        hf_id="mteb/stsbenchmark-sts",
        prompt_key="stsbenchmark_mteb",
        score_range=(0.0, 5.0),
        parse_pattern=r"\b\d+(?:\.\d+)?\b",
        description="MTEB STS Benchmark scored on 0-5.",
    ),
    "sickr_sts": DatasetMeta(
        key="sickr_sts",
        hf_id="Samsoup/sickr-sts",
        prompt_key="sickr_sts",
        score_range=(1.0, 5.0),
        parse_pattern=r"\b\d+(?:\.\d+)?\b",
        description="SICK-R STS scored on 1-5.",
    ),
    "sts22": DatasetMeta(
        key="sts22",
        hf_id="Samsoup/sts22-crosslingual-sts",
        prompt_key="sts22",
        score_range=(1.0, 4.0),
        parse_pattern=r"\b\d+(?:\.\d+)?\b",
        description="STS22 cross-lingual scored on 1-4.",
    ),
    "wmt_en_ru": DatasetMeta(
        key="wmt_en_ru",
        hf_id="samsoup/Samsoup-WMT2020-ru-en",
        prompt_key="wmt_en_ru",
        score_range=(0.0, 100.0),
        parse_pattern=r"(-?\d+(?:\.\d+)?)",
        description="WMT20 Russian→English scored on 0-100.",
    ),
    "wmt_en_zh": DatasetMeta(
        key="wmt_en_zh",
        hf_id="samsoup/Samsoup-WMT2020-en-zh",
        prompt_key="wmt_en_zh",
        score_range=(0.0, 100.0),
        parse_pattern=r"(-?\d+(?:\.\d+)?)",
        description="WMT20 English→Chinese scored on 0-100.",
    ),
    "wmt_si_en": DatasetMeta(
        key="wmt_si_en",
        hf_id="samsoup/Samsoup-WMT2020-si-en",
        prompt_key="wmt_si_en",
        score_range=(0.0, 100.0),
        parse_pattern=r"(-?\d+(?:\.\d+)?)",
        description="WMT20 Sinhala→English scored on 0-100.",
    ),
}


# =========================
# Aliases + key lists
# =========================

PAIR_DATASET_ALIASES: Dict[str, str] = {
    # stsbenchmark
    "stsbenchmark": "stsbenchmark_mteb",
    "stsbenchmteb": "stsbenchmark_mteb",
    "stsbenchmarkmteb": "stsbenchmark_mteb",
    # sickr
    "sickr": "sickr_sts",
    "sickrsts": "sickr_sts",
    "sickr_sts": "sickr_sts",
    # sts22
    "sts22crosslingualsts": "sts22",
    "sts22_crosslingual_sts": "sts22",
    "sts22xlingsts": "sts22",
    "sts22_xling_sts": "sts22",
    # wmt ru-en
    "wmt20ruen": "wmt_en_ru",
    "wmt20_ru_en": "wmt_en_ru",
    "wmt20enru": "wmt_en_ru",
    "wmt_en_ru": "wmt_en_ru",
    "wmtenru": "wmt_en_ru",
    "wmtenru": "wmt_en_ru",
    # wmt en-zh
    "wmt20enzh": "wmt_en_zh",
    "wmt20_en_zh": "wmt_en_zh",
    "wmt20enzh": "wmt_en_zh",
    "wmt_en_zh": "wmt_en_zh",
    "wmtenzh": "wmt_en_zh",
    # wmt si-en
    "wmt20sien": "wmt_si_en",
    "wmt20_si_en": "wmt_si_en",
    "wmt20ensi": "wmt_si_en",
    "wmt_si_en": "wmt_si_en",
    "wmtsien": "wmt_si_en",
    # stsb
    "stsb": "stsb",
}

PAIR_DATASET_KEYS = sorted(DATASET_METAS.keys())
PAIR_DATASET_ALL_KEYS = sorted(
    set(DATASET_METAS.keys()) | set(PAIR_DATASET_ALIASES.keys())
)


# =========================
# Helpers (public API)
# =========================

DatasetRef = Union[str, DatasetMeta]


def canonicalize_dataset_name(name: Optional[str]) -> Optional[str]:
    """
    Canonicalize user input to a stable alias key:
    - lowercase
    - remove spaces, underscores, hyphens
    """
    if not name:
        return None
    return re.sub(r"[\s_-]+", "", name.strip().lower())


def resolve_dataset_key(name: Optional[str]) -> Optional[str]:
    """
    Resolve a user-provided dataset name/alias to a canonical key in DATASET_METAS.
    Returns None if unknown.
    """
    c = canonicalize_dataset_name(name)
    if c is None:
        return None
    if c in DATASET_METAS:
        return c
    return PAIR_DATASET_ALIASES.get(c)


def get_dataset_meta(name: Optional[str]) -> Optional[DatasetMeta]:
    key = resolve_dataset_key(name)
    return DATASET_METAS.get(key) if key else None


def get_prompt_template(name: Optional[str]) -> Optional[PromptTemplate]:
    meta = get_dataset_meta(name)
    return PROMPT_TEMPLATES.get(meta.prompt_key) if meta else None


def format_score(x: float) -> str:
    return f"{float(x):.3f}"


def clamp_and_round(
    value: float, bounds: Tuple[float, float], decimals: Optional[int]
) -> float:
    value = max(bounds[0], min(bounds[1], float(value)))
    return float(f"{value:.{decimals}f}") if decimals is not None else value


def parse_score(raw_answer: str, dataset: DatasetRef) -> float:
    """
    Parse numeric score from model output, then clamp+round to dataset bounds.
    Uses dataset-specific regex to allow some robustness to formatting drift.
    """
    meta = (
        dataset
        if isinstance(dataset, DatasetMeta)
        else get_dataset_meta(dataset)
    )
    if meta is None:
        raise ValueError(f"Unknown dataset for parsing: {dataset}")

    matches = re.findall(meta.parse_pattern, raw_answer.strip())
    if not matches:
        raise ValueError(f"Could not parse numeric score from: {raw_answer!r}")

    last = matches[-1]
    if isinstance(last, tuple):  # for patterns with groups
        last = last[0]

    score = float(last)
    return clamp_and_round(score, meta.score_range, meta.round_predictions)
