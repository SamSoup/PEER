# Data module layout

This package standardizes regression datasets into a small set of building blocks.

- `base.py` holds the generic `BaseRegressionDataModule` plus two specializations:
  - `PairSentenceRegressionDataModule` (currently used) for sentence–sentence → score tasks with optional field combination and tokenization controls.
  - `SingleSentenceRegressionDataModule` (skeleton) for future single-text → score datasets.
- `pairwise_dataset_modules.py` contains the registry of supported pairwise datasets, alias normalization, and `build_pairwise_datamodule(...)` to instantiate a module from the registry.
- `factory.py` currently forwards `get_datamodule(...)` to the pairwise registry builder.
- `examples.py` shows how to instantiate datamodules programmatically for the supported pairwise datasets.

## Implemented pairwise datasets
Canonical names (aliases shown in code) mapped to HF sources and label ranges:

- `stsb` → `sentence-transformers/stsb` (label_max=1.0)
- `stsbenchmark-mteb` → `mteb/stsbenchmark-sts` (label_max=5.0)
- `sickr-sts` → `samsoup/sickr-sts` (label_max=5.0)
- `sts22-crosslingual-sts` → `Samsoup/sts22-crosslingual-sts` (label_max=4.0)
- `wmt20-en-zh` → `samsoup/Samsoup-WMT2020-en-zh` (label_max=100.0)
- `wmt20-ru-en` → `samsoup/Samsoup-WMT2020-ru-en` (label_max=100.0)
- `wmt20-si-en` → `samsoup/Samsoup-WMT2020-si-en` (label_max=100.0)

Add new pairwise datasets by extending `DATASET_METAS` (and optional `DATASET_ALIASES`) in `data/datasets.py`.

## Future single-sentence datasets
When adding one-sentence → score datasets, create a small registry/builder around `SingleSentenceRegressionDataModule` and update `get_datamodule(...)` to route to it. A new helper file mirroring `pairwise_dataset_modules.py` will keep that logic isolated.
