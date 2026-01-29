# data/lm.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.pair_dataset_registry import (
    clamp_and_round,
    format_score,
    get_dataset_meta,
    get_prompt_template,
)


class PromptMaskedLMCollator:
    """
    Pads input_ids/attention_mask using tokenizer.pad and pads labels with -100
    on the SAME side as tokenizer.padding_side.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]  # list[list[int]]

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = int(batch["input_ids"].shape[1])

        left = self.tokenizer.padding_side == "left"
        padded_labels: List[List[int]] = []
        for lab in labels:
            pad_len = max_len - len(lab)
            if pad_len < 0:
                lab = lab[-max_len:]
                pad_len = 0
            padded_labels.append(
                ([-100] * pad_len + lab) if left else (lab + [-100] * pad_len)
            )

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class PromptMaskedCausalLMDataModule(pl.LightningDataModule):
    """
    Registry-only causal LM datamodule for pair datasets.

    Builds prompt = (system + pair + score_instruction) and completion = gold score string.
    Tokenizes prompt and completion separately, then packs into:
      input_ids = prompt_ids + completion_ids (+ eos)
      labels    = [-100]*len(prompt_ids) + completion_ids (+ eos)
    Then left-truncates to max_seq_length (keeps completion tokens).
    """

    def __init__(
        self,
        *,
        dataset_key: str,
        model_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 4,
        eval_batch_size: int = 4,
        combine_fields: bool = False,
        combine_separator: Optional[str] = None,  # None => auto sep/eos/\n\n
        add_eos: bool = True,
        use_chat_template_if_available: bool = True,
        # map knobs
        map_batch_size: int = 256,
        map_num_proc: Optional[int] = None,
        load_from_cache_file: bool = True,
        keep_in_memory: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset_key = dataset_key
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = int(max_seq_length)
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size)

        self.combine_fields = bool(combine_fields)
        self.combine_separator = combine_separator
        self.add_eos = bool(add_eos)
        self.use_chat_template_if_available = bool(
            use_chat_template_if_available
        )

        self.map_batch_size = int(map_batch_size)
        self.map_num_proc = map_num_proc
        self.load_from_cache_file = bool(load_from_cache_file)
        self.keep_in_memory = bool(keep_in_memory)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.meta = get_dataset_meta(dataset_key)
        self.template = get_prompt_template(dataset_key)
        if self.meta is None or self.template is None:
            raise ValueError(f"Unsupported dataset_key/alias: {dataset_key}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self._ensure_pad_token()

        # left-padding works well with prompt-only generation in your baseline
        self.tokenizer.padding_side = "left"

        self.collate_fn = PromptMaskedLMCollator(self.tokenizer)

        self.dataset: Optional[datasets.DatasetDict] = None
        self._prepared: Dict[str, bool] = {}

    # -------- lifecycle --------

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = datasets.load_dataset(
                self.meta.hf_id,
                self.meta.dataset_config_name,
                **(self.meta.load_dataset_kwargs or {}),
            )

        for split in self._splits_for_stage(stage):
            if self._prepared.get(split):
                continue
            self.dataset[split] = self._map_split(split)
            self._prepared[split] = True

    def _splits_for_stage(self, stage: Optional[str]) -> List[str]:
        stage = (stage or "fit").lower()
        if stage == "fit":
            return ["train", "validation"]
        if stage == "validate":
            return ["validation"]
        if stage in ("test", "predict"):
            return ["test"]
        return ["train", "validation", "test"]

    # -------- dataloaders --------

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    # -------- prompt + packing --------

    def _ensure_pad_token(self):
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _build_user_text(self, s1: str, s2: str) -> str:
        pair_part = self.template.pair_intro.format(
            sentence_a=s1.strip(), sentence_b=s2.strip()
        )
        return f"{pair_part}{self.template.score_instruction}\n"

    def _build_completion_text(self, raw_score: Any) -> str:
        score = clamp_and_round(
            float(raw_score), self.meta.score_range, self.meta.round_predictions
        )
        return format_score(score)

    def _as_ids(self, tok_out: Any) -> List[int]:
        """
        Normalize tokenizer outputs to a flat List[int] (single example).
        Handles: list[int], dict/BatchEncoding with input_ids, torch tensors, nested lists.
        """
        if tok_out is None:
            return []
        # already a list of ints
        if isinstance(tok_out, list):
            if not tok_out:
                return []
            # nested list (batch of 1)
            if isinstance(tok_out[0], list):
                return [int(x) for x in tok_out[0]]
            return [int(x) for x in tok_out]

        # BatchEncoding / dict-like
        if isinstance(tok_out, dict) and "input_ids" in tok_out:
            return self._as_ids(tok_out["input_ids"])

        # torch tensor
        if isinstance(tok_out, torch.Tensor):
            if tok_out.dim() == 2:  # batch of 1 likely
                tok_out = tok_out[0]
            return [int(x) for x in tok_out.detach().cpu().tolist()]

        # last resort: try attribute access
        ids = getattr(tok_out, "input_ids", None)
        if ids is not None:
            return self._as_ids(ids)

        raise TypeError(
            f"Cannot normalize token output to ids. Got type={type(tok_out)}"
        )

    def _prompt_ids(self, user_text: str) -> List[int]:
        # Chat-template path (preferred if present): do NOT duplicate system inside user.
        if self.use_chat_template_if_available and getattr(
            self.tokenizer, "chat_template", None
        ):
            msgs = [
                {"role": "system", "content": self.template.system},
                {"role": "user", "content": user_text},
            ]
            out = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,  # keep python lists / BatchEncoding (normalized below)
                enable_thinking=False,  # This stops the <think> token injection
            )
            return self._as_ids(out)

        # Plain-text path
        prompt_text = f"{self.template.system}\n\n{user_text}"
        out = self.tokenizer(prompt_text, add_special_tokens=False)
        return self._as_ids(out)

    def _completion_ids(self, completion_text: str) -> List[int]:
        out = self.tokenizer(completion_text, add_special_tokens=False)
        return self._as_ids(out)

    def _pack_left_truncate(
        self, prompt_ids: List[int], completion_ids: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        eos = self.tokenizer.eos_token_id
        ids = (
            prompt_ids
            + completion_ids
            + ([int(eos)] if (self.add_eos and eos is not None) else [])
        )

        if len(ids) <= self.max_seq_length:
            kept_ids = ids
            kept_prompt = len(prompt_ids)
        else:
            overflow = len(ids) - self.max_seq_length
            kept_ids = ids[
                -self.max_seq_length :
            ]  # keep tail (completion-biased)
            kept_prompt = max(0, len(prompt_ids) - overflow)

        labels = ([-100] * kept_prompt) + kept_ids[kept_prompt:]
        attn = [1] * len(kept_ids)
        return kept_ids, attn, labels

    def _map_split(self, split: str):
        assert self.dataset is not None
        s1f, s2f, yfield = (
            self.meta.sentence1_field,
            self.meta.sentence2_field,
            self.meta.score_field,
        )

        def fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            input_ids: List[List[int]] = []
            attention_mask: List[List[int]] = []
            labels: List[List[int]] = []

            for s1, s2, y in zip(batch[s1f], batch[s2f], batch[yfield]):
                user_text = self._build_user_text(str(s1), str(s2))
                completion = self._build_completion_text(y)

                p_ids = self._prompt_ids(user_text)
                c_ids = self._completion_ids(completion)

                ids, attn, lab = self._pack_left_truncate(p_ids, c_ids)
                input_ids.append(ids)
                attention_mask.append(attn)
                labels.append(lab)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        remove_cols = self.dataset[split].column_names
        return self.dataset[split].map(
            fn,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.map_num_proc,
            remove_columns=remove_cols,
            load_from_cache_file=self.load_from_cache_file,
            keep_in_memory=self.keep_in_memory,
            desc=f"lm[{split}]",
        )
