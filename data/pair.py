# /data/pair.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import datasets
from data.base import BaseRegressionDataModule
from data.pair_dataset_registry import (
    get_dataset_meta,
)


def clamp_and_round(
    x: float, bounds: Tuple[float, float], decimals: Optional[int]
) -> float:
    x = max(bounds[0], min(bounds[1], float(x)))
    return float(f"{x:.{decimals}f}") if decimals is not None else x


class PairSentenceRegressionDataModule(BaseRegressionDataModule):
    """
    Two explicit paths:
      A) raw text:  returns {"text": str | (str,str), "labels": float}
      B) tokenized: returns token dict + {"labels": float}

    Combined vs not-combined is handled inside the SAME map pass (no extra columns).
    """

    def __init__(
        self,
        *,
        dataset_key: str,  # <-- registry key or alias
        model_name_or_path: str,
        combine_fields: bool = False,
        combine_separator: Optional[
            str
        ] = None,  # None => auto sep token / eos / "\n\n"
        tokenize_inputs: bool = True,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        map_batch_size: int = 1024,
        map_num_proc: Optional[int] = None,  # None = single-process, safest
        load_from_cache_file: bool = True,
        keep_in_memory: bool = False,
    ):
        self.meta = get_dataset_meta(dataset_key)
        if self.meta is None:
            raise ValueError(f"Unknown dataset_key/alias: {dataset_key}")

        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenize_inputs=tokenize_inputs,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.combine_fields = combine_fields
        self.combine_separator = combine_separator
        self.map_batch_size = map_batch_size
        self.map_num_proc = map_num_proc
        self.load_from_cache_file = load_from_cache_file
        self.keep_in_memory = keep_in_memory

    # ---------- stage-aware setup ----------
    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = datasets.load_dataset(
                self.meta.hf_id,
                self.meta.dataset_config_name,
                **(self.meta.load_dataset_kwargs or {}),
            )

        # Only prepare needed splits per stage
        needed = self._splits_for_stage(stage)
        for split in needed:
            self._prepare_split(split)

    def _splits_for_stage(self, stage: Optional[str]) -> List[str]:
        stage = (stage or "fit").lower()
        if stage == "fit":
            return ["train", "validation"]
        if stage == "validate":
            return ["validation"]
        if stage in ("test", "predict"):
            return ["test"]
        # fallback: be safe
        return ["train", "validation", "test"]

    # ---------- split preparation ----------
    def _prepare_split(self, split: str):
        mode = "tok" if self.tokenize_inputs else "raw"
        if self._prepared.get(split) == mode:
            return

        if mode == "tok":
            self.dataset[split] = self._map_tokenized(split)
        else:
            self.dataset[split] = self._map_raw(split)

        self._prepared[split] = mode

    # ---------- separators / labels ----------
    def _sep(self) -> str:
        if self.combine_separator is not None:
            return f" {self.combine_separator} "
        if self.tokenizer is not None and getattr(
            self.tokenizer, "sep_token", None
        ):
            return f" {self.tokenizer.sep_token} "
        if self.tokenizer is not None and getattr(
            self.tokenizer, "eos_token", None
        ):
            return f" {self.tokenizer.eos_token} "
        return "\n\n"

    def _label_list(self, batch: Dict[str, List[Any]]) -> List[float]:
        xs = batch[self.meta.score_field]
        return [
            clamp_and_round(
                x, self.meta.score_range, self.meta.round_predictions
            )
            for x in xs
        ]

    # ---------- path A: raw ----------
    def _map_raw(self, split: str):
        s1, s2 = self.meta.sentence1_field, self.meta.sentence2_field
        sep = self._sep()

        def fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            if self.combine_fields:
                text = [a + sep + b for a, b in zip(batch[s1], batch[s2])]
            else:
                text = list(zip(batch[s1], batch[s2]))
            return {"text": text, "labels": self._label_list(batch)}

        # remove everything except what we emit
        remove_cols = [
            c
            for c in self.dataset[split].column_names
            if c not in (s1, s2, self.meta.score_field)
        ]
        out = self.dataset[split].map(
            fn,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.map_num_proc,
            remove_columns=remove_cols,
            load_from_cache_file=self.load_from_cache_file,
            keep_in_memory=self.keep_in_memory,
            desc=f"raw[{split}]",
        )
        return out

    # ---------- path B: tokenized ----------
    def _map_tokenized(self, split: str):
        assert self.tokenizer is not None
        s1, s2 = self.meta.sentence1_field, self.meta.sentence2_field
        sep = self._sep()

        def fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            if self.combine_fields:
                texts = [a + sep + b for a, b in zip(batch[s1], batch[s2])]
                enc = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,  # dynamic padding in collator
                    return_tensors=None,
                )
            else:
                enc = self.tokenizer(
                    batch[s1],
                    batch[s2],
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,  # dynamic padding in collator
                    return_tensors=None,
                )
            enc["labels"] = self._label_list(batch)
            return enc

        remove_cols = self.dataset[split].column_names
        out = self.dataset[split].map(
            fn,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.map_num_proc,
            remove_columns=remove_cols,
            load_from_cache_file=self.load_from_cache_file,
            keep_in_memory=self.keep_in_memory,
            desc=f"tok[{split}]",
        )
        return out
