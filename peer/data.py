from __future__ import annotations

from torch.utils.data import DataLoader

from data.factory import build_pairwise_datamodule
from peer.prompting import build_prompts


def build_raw_dataloaders(
    *,
    dataset_name: str,  # registry key or alias
    model_name: str,  # tokenizer name (only used to pick sep token if needed)
    batch_size: int = 16,
    max_length: int = 256,
    combine_fields: bool = False,
    combine_separator_token: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Instantiate PairSentenceRegressionDataModule and return raw-text train/val loaders.

    Uses the new registry-driven datamodule builder, but forces tokenize_inputs=False so
    examples stay as raw text:
      - combined: {"text": str, "labels": float}
      - not combined: {"text": (str,str), "labels": float}
    """
    dm = build_pairwise_datamodule(
        dataset_name=dataset_name,
        model_name=model_name,
        max_seq_length=max_length,  # unused by raw path but kept consistent
        batch_size=batch_size,
        tokenize_inputs=False,
        combine_fields=combine_fields,
        combine_separator_token=combine_separator_token,
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    dm.train_batch_size = batch_size
    dm.eval_batch_size = batch_size
    dm.num_workers = num_workers
    dm.pin_memory = pin_memory

    dm.setup("fit")

    train_loader = DataLoader(
        dm.dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dm._collate_raw,
    )
    val_loader = DataLoader(
        dm.dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dm._collate_raw,
    )

    return train_loader, val_loader, dm


def prepare_prompts(texts, dataset_name: str | None):
    """Convert raw texts or pairs into dataset-specific prompts."""
    return build_prompts(texts, dataset_name)


def build_cache_loader(
    dm, batch_size: int, *, num_workers: int = 0, pin_memory: bool = True
):
    """Non-shuffled loader over the training split for cache construction."""
    dm.setup("fit")
    return DataLoader(
        dm.dataset["train"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dm._collate_raw,
    )


def build_test_loader(
    dm, batch_size: int, *, num_workers: int = 0, pin_memory: bool = True
):
    """Loader over the test split for evaluation."""
    dm.setup("test")
    return DataLoader(
        dm.dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dm._collate_raw,
    )
