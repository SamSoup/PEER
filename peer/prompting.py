"""
Prompt helpers that mirror the dataset->prompt mapping in data/datasets.py.
Builds a full prompt string (system + user message) for a given dataset key.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

from data.pair_dataset_registry import PROMPT_TEMPLATES, get_dataset_meta


RawText = Union[str, Tuple[str, str], List[str], List[Tuple[str, str]]]


DEFAULT_TMPL = "Input:\n{text}\n\nPredict a score from 1 to 4:"


def prompt_key_for_dataset(dataset_name: str | None) -> str | None:
    meta = get_dataset_meta(dataset_name)
    return meta.prompt_key if meta else None


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


def build_prompt_for_example(
    text: RawText, dataset_name: str | None = None
) -> str:
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


def build_prompts(
    texts: Iterable[RawText], dataset_name: str | None = None
) -> List[str]:
    """Vectorized helper over a batch of raw texts."""
    return [build_prompt_for_example(t, dataset_name) for t in texts]
