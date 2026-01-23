# data/factory.py

from .pairwise_dataset_modules import build_pairwise_datamodule


def get_datamodule(
    dataset_name: str,
    model_name: str,
    max_seq_length: int,
    batch_size: int,
    *,
    tokenize_inputs: bool = True,
    combine_fields: bool = False,
    combine_separator_token: str = "[SEP]",
):
    """
    Returns an initialized LightningDataModule for the given dataset name.
    """
    dm = build_pairwise_datamodule(
        dataset_name=dataset_name,
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        tokenize_inputs=tokenize_inputs,
        combine_fields=combine_fields,
        combine_separator_token=combine_separator_token,
    )
    if dm is not None:
        return dm

    raise ValueError(f"Unsupported dataset name: {dataset_name}")
