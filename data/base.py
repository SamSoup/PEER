# /data/base.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding


class RegressionTokenCollator:
    """Pads tokenized batches and ensures float32 labels for regression."""

    def __init__(self, tokenizer, max_length: int):
        self.inner = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.inner(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.float32)
        return batch


class BaseRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenize_inputs = tokenize_inputs
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = None
        self.collate_fn = None

        if self.tokenize_inputs:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.collate_fn = RegressionTokenCollator(
                self.tokenizer, max_length=max_seq_length
            )

        # subclasses fill these
        self.dataset = None  # expected DatasetDict with train/validation/test
        self._prepared: Dict[str, str] = (
            {}
        )  # split -> mode marker ("raw" or "tok")

    # dataloaders
    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=(
                self.collate_fn if self.tokenize_inputs else self._collate_raw
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=(
                self.collate_fn if self.tokenize_inputs else self._collate_raw
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=(
                self.collate_fn if self.tokenize_inputs else self._collate_raw
            ),
        )

    # raw collate (expects {"text": ..., "labels": ...})
    def _collate_raw(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [ex["text"] for ex in batch]
        labels = torch.tensor(
            [float(ex["labels"]) for ex in batch], dtype=torch.float32
        )
        return {"text": texts, "labels": labels}
