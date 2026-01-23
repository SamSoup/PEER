from torch.utils.data import DataLoader

from data.factory import get_datamodule
from peer.prompting import build_prompts


def _resolve_val_split(dm):
    if getattr(dm, "eval_splits", None):
        return dm.eval_splits[0] if dm.eval_splits else None
    if dm.dataset and "validation" in dm.dataset:
        return "validation"
    return None


def build_raw_dataloaders(
    *,
    dataset_name: str,
    model_name: str,
    batch_size: int = 16,
    max_length: int = 256,
    combine_fields: bool = False,
):
    """
    Instantiate datamodule from /data and return raw-text train/val dataloaders.
    """
    dm = get_datamodule(
        dataset_name=dataset_name,
        model_name=model_name,
        max_seq_length=max_length,
        batch_size=batch_size,
        tokenize_inputs=False,
        combine_fields=combine_fields,
    )
    dm.setup("fit")
    collate = dm._collate_raw
    train_loader = DataLoader(
        dm.dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_split = _resolve_val_split(dm)
    val_loader = None
    if val_split:
        val_loader = DataLoader(
            dm.dataset[val_split],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
    return train_loader, val_loader, dm


def prepare_prompts(texts, dataset_name: str | None):
    """Convert raw texts or pairs into dataset-specific prompts."""
    return build_prompts(texts, dataset_name)


def build_cache_loader(dm, batch_size: int):
    """Non-shuffled loader over the training split for cache construction."""
    return DataLoader(
        dm.dataset["train"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dm._collate_raw,
    )


def build_test_loader(dm, batch_size: int):
    """Loader over the test split for evaluation."""
    if "test" not in dm.dataset:
        raise ValueError("Datamodule does not have a test split.")
    return DataLoader(
        dm.dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dm._collate_raw,
    )
