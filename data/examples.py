# Example usage of the pairwise regression datamodules.
# This file does not run downloads unless invoked directly.

from __future__ import annotations

import argparse
from typing import Iterable

from data.factory import get_datamodule
from data.pairwise_dataset_modules import (
    PAIRWISE_DATASETS,
    PAIRWISE_NAME_ALIASES,
    build_pairwise_datamodule,
)


def iter_dataset_names() -> Iterable[str]:
    """Yield canonical names for the registered pairwise datasets."""
    return PAIRWISE_DATASETS.keys()


def demo_via_factory(dataset_name: str, model_name: str):
    """
    Instantiate through the public get_datamodule() helper.
    Uses alias normalization inside the factory.
    """
    dm = get_datamodule(
        dataset_name=dataset_name,
        model_name=model_name,
        max_seq_length=128,
        batch_size=16,
        tokenize_inputs=True,
        combine_fields=False,
    )
    dm.setup("fit")
    print(f"[factory] Loaded {dataset_name}: splits={list(dm.dataset.keys())}")


def demo_direct_build(dataset_name: str, model_name: str):
    """
    Instantiate directly from the registry builder.
    Useful if you want to inspect or tweak fields before training.
    """
    dm = build_pairwise_datamodule(
        dataset_name=dataset_name,
        model_name=model_name,
        max_seq_length=128,
        batch_size=16,
        tokenize_inputs=True,
        combine_fields=False,
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dm.setup("fit")
    print(
        f"[direct] Loaded {dataset_name} (canonical={dm.dataset_name}): "
        f"splits={list(dm.dataset.keys())}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Instantiate pairwise regression datamodules."
    )
    parser.add_argument(
        "--dataset",
        default="stsb",
        help="Canonical name or alias (e.g., stsb, sts22, wmt20-en-zh).",
    )
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="HF model name/path for tokenization.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported dataset names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Canonical pairwise datasets:")
        for name in iter_dataset_names():
            hf_source = PAIRWISE_DATASETS[name]["dataset_name"]
            label_max = PAIRWISE_DATASETS[name].get("label_max", 1.0)
            print(f"  - {name:22s} -> {hf_source} (label_max={label_max})")
        if PAIRWISE_NAME_ALIASES:
            print("\nAliases:")
            for alias, canonical in sorted(PAIRWISE_NAME_ALIASES.items()):
                print(f"  - {alias:22s} -> {canonical}")
        return

    print("Instantiating via factory...")
    demo_via_factory(dataset_name=args.dataset, model_name=args.model)

    print("Instantiating directly from registry...")
    demo_direct_build(dataset_name=args.dataset, model_name=args.model)


if __name__ == "__main__":
    main()
