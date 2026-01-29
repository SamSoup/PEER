from data.base import BaseRegressionDataModule


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
