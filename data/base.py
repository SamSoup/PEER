# /data/base.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import datasets


class BaseRegressionDataModule(pl.LightningDataModule):
    """
    Generic regression DataModule that can:
      - tokenize (HuggingFace tokenizer) OR pass raw text downstream
      - handle single-field or pair-field tokenization (via self.text_fields set
      in subclass/setup)

    Subclasses must:
      - load `self.dataset` with splits: train/validation/test
      - define `self.text_fields`:
          * ["combined_text"]           -> single string per example
          * ["sentence1", "sentence2"]  -> a pair per example
      - call `_tokenize_splits(remove_columns=[...])` if tokenize_inputs=True
      - OR create raw fields: "text" (string or (string,string)) and the label
        column indicated by `self.output_column` (default: "label"). Collation
        will emit "labels".
    """

    # Columns used when tokenizing with HF tokenizer
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        output_column: str = "label",  # name of label column in the source dataset
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenize_inputs = tokenize_inputs
        self.output_column = output_column

        # Optional metadata helpers used by downstream pipeline stages
        self.label_max = 1.0

        self.tokenizer = None
        if self.tokenize_inputs:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=True
            )
            # Some causal LLMs lack an explicit pad token; align padding to EOS.
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # Fallback: add a new PAD token if EOS is also missing.
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if (
                self.tokenizer.pad_token_id is None
                and self.tokenizer.eos_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if not self.tokenizer.padding_side:
                self.tokenizer.padding_side = "right"

        self.dataset = None
        self.columns = None
        self.eval_splits = []
        # Subclass should set this during setup() to either:
        #   ["combined_text"] OR ["sentence1", "sentence2"]
        self.text_fields = None

    def setup(self, stage: str = None):
        raise NotImplementedError(
            "Each dataset-specific DataModule must override `setup()`"
        )

    # -------- dataloaders --------
    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=None if self.tokenize_inputs else self._collate_raw,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            collate_fn=None if self.tokenize_inputs else self._collate_raw,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            collate_fn=None if self.tokenize_inputs else self._collate_raw,
        )

    # -------- helpers --------
    def _collate_raw(self, batch):
        """
        Collate raw text examples (no tokenization).
        Expects each item to have:
          - "text": str OR tuple(str, str)
          - label under either "labels" or `self.output_column`
        Returns:
          { "text": List[str] or List[Tuple[str,str]], "labels": Tensor[B] }
        """
        texts = [ex["text"] for ex in batch]
        # accept either normalized "labels" or original dataset column name
        labels_list = []
        for ex in batch:
            if "labels" in ex:
                labels_list.append(float(ex["labels"]))
            else:
                labels_list.append(float(ex[self.output_column]))
        labels = torch.tensor(labels_list, dtype=torch.float32)
        return {"text": texts, "labels": labels}

    def _tokenize_splits(self, remove_columns):
        """
        Tokenize using self.text_fields:
        - If len(self.text_fields) == 1: tokenizes a single string field (e.g., "combined_text")
        - If len(self.text_fields) == 2: tokenizes a pair (sentence1, sentence2)

        Safe on HPC: single-process map, no Arrow cache reuse, keep results in RAM.
        """
        if not self.tokenize_inputs:
            raise RuntimeError(
                "Called _tokenize_splits while tokenize_inputs=False."
            )
        if not self.text_fields:
            raise RuntimeError(
                "self.text_fields must be set before calling _tokenize_splits()."
            )
        if len(self.text_fields) not in (1, 2):
            raise ValueError("self.text_fields must have length 1 or 2.")

        # reset in case setup() is called multiple times
        self.eval_splits = []

        # Ensure we’re in “python” mode before mapping to avoid Arrow formatting pulls
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].with_format("python")

        common_map_kwargs = dict(
            batched=True,
            batch_size=1024,  # tune if RAM is tight
            num_proc=0,  # <<< critical: disable multiprocessing to avoid CUDA/fork issues
            load_from_cache_file=False,  # <<< do not reuse old Arrow cache
            keep_in_memory=True,  # <<< build new table fully in RAM
            desc="Map",
        )

        # Map safely
        new_splits = {}
        for split in self.dataset.keys():
            print(f"Processing split: {split}")
            new_splits[split] = self.dataset[split].map(
                self._tokenize,
                remove_columns=remove_columns,
                **common_map_kwargs,
            )
            # After mapping, set torch format on the exact columns we’ll read
            self.columns = [
                c
                for c in new_splits[split].column_names
                if c in self.loader_columns
            ]
            print(f"Using columns: {self.columns}")
            new_splits[split].set_format(
                type="torch", columns=self.columns, output_all_columns=False
            )
            if "validation" in split:
                self.eval_splits.append(split)

        self.dataset = new_splits

    def _tokenize(self, example_batch, indices=None):
        # Build batch as either a list[str] or a list[Tuple[str,str]]
        if len(self.text_fields) == 1:
            texts_or_text_pairs = example_batch[self.text_fields[0]]
        else:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )

        # IMPORTANT: return_tensors=None -> plain Python lists, safe for datasets.map
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            return_tensors=None,  # <<< was "pt"; change to None
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )

        # Normalize the dataset's label column name -> "labels" (as a list of floats)
        # example_batch[self.output_column] is already a list; ensure float conversion:
        labels_list = [float(x) for x in example_batch[self.output_column]]
        features["labels"] = labels_list

        return features

    def _load_dataset(self):
        """
        Helper to load the configured dataset using datasets.load_dataset().
        Subclasses must define:
          - self.dataset_name (str)
          - optional: self.dataset_config_name (str | None)
          - optional: self.load_dataset_kwargs (dict)
        """
        if not hasattr(self, "dataset_name"):
            raise AttributeError(
                "Subclasses must set `self.dataset_name` before calling _load_dataset()."
            )

        load_args = [self.dataset_name]
        dataset_config_name = getattr(self, "dataset_config_name", None)
        if dataset_config_name:
            load_args.append(dataset_config_name)

        load_dataset_kwargs = getattr(self, "load_dataset_kwargs", {}) or {}
        return datasets.load_dataset(*load_args, **load_dataset_kwargs)


class SingleSentenceRegressionDataModule(BaseRegressionDataModule):
    """
    Base class for single-sentence regression datasets (text -> scalar score).
    Provides:
      - tokenized or raw-text pathways
      - optional label_max metadata
    """

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        *,
        text_field: str = "text",
        output_column: str = "label",
        dataset_config_name: str | None = None,
        load_dataset_kwargs: dict | None = None,
        label_max: float | None = None,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenize_inputs=tokenize_inputs,
            output_column=output_column,
            **kwargs,
        )
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.load_dataset_kwargs = load_dataset_kwargs or {}
        self.text_field = text_field
        if label_max is not None:
            self.label_max = label_max

    def setup(self, stage: str | None = None):
        # load dataset and reset eval_splits for re-entrancy
        self.dataset = self._load_dataset()
        self.eval_splits = []

        if self.tokenize_inputs:
            self.text_fields = [self.text_field]
            self._tokenize_splits(remove_columns=[self.output_column])
        else:
            # raw text pathway
            for split in self.dataset:
                self.dataset[split] = self.dataset[split].map(
                    lambda x: {
                        "text": x[self.text_field],
                        "labels": x[self.output_column],
                    }
                )

            for split in self.dataset:
                self.dataset[split].set_format(type=None)
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)


class PairSentenceRegressionDataModule(BaseRegressionDataModule):
    """
    Base class for sentence-pair regression datasets.
    Handles:
      - optional field combination with a separator token
      - tokenized or raw-text pathways
      - optional label_max metadata for downstream components
    """

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        *,
        sentence1_field: str = "sentence1",
        sentence2_field: str = "sentence2",
        output_column: str = "score",
        dataset_config_name: str | None = None,
        load_dataset_kwargs: dict | None = None,
        label_max: float | None = None,
        combine_fields: bool = False,
        combine_separator_token: str = "[SEP]",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenize_inputs=tokenize_inputs,
            output_column=output_column,
            **kwargs,
        )
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.load_dataset_kwargs = load_dataset_kwargs or {}
        self.sentence1_field = sentence1_field
        self.sentence2_field = sentence2_field
        self.combine_fields = combine_fields
        self.combine_separator_token = combine_separator_token
        if label_max is not None:
            self.label_max = label_max

    def setup(self, stage: str | None = None):
        # load dataset and reset eval_splits for re-entrancy
        self.dataset = self._load_dataset()
        self.eval_splits = []

        if self.tokenize_inputs:
            if self.combine_fields:
                for split in self.dataset:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "combined_text": x[self.sentence1_field]
                            + f" {self.combine_separator_token} "
                            + x[self.sentence2_field],
                            self.output_column: x[self.output_column],
                        }
                    )
                self.text_fields = ["combined_text"]
            else:
                self.text_fields = [self.sentence1_field, self.sentence2_field]

            self._tokenize_splits(remove_columns=[self.output_column])
        else:
            for split in self.dataset:
                if self.combine_fields:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": x[self.sentence1_field]
                            + f" {self.combine_separator_token} "
                            + x[self.sentence2_field],
                            "labels": x[self.output_column],
                        }
                    )
                else:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": (
                                x[self.sentence1_field],
                                x[self.sentence2_field],
                            ),
                            "labels": x[self.output_column],
                        }
                    )

            for split in self.dataset:
                self.dataset[split].set_format(type=None)
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)
