# /baselines/SeqCls.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification

import torchmetrics


_HEAD_ATTRS = (
    "classifier",
    "score",
    "classification_head",
    "regression_head",
    "lm_head",
    "pre_classifier",
)


class SequenceClassificationRegressor(pl.LightningModule):
    """
    Regression via AutoModelForSequenceClassification(num_labels=1).

    If freeze_encoder=True:
      - freeze all params
      - unfreeze known head modules (classifier/score/...)
    """

    def __init__(
        self,
        model_name: str,
        lr: float = 3e-5,
        weight_decay: float = 0.0,
        freeze_encoder: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        config.num_labels = 1
        config.problem_type = "regression"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

        if freeze_encoder:
            self._freeze_all()
            self._unfreeze_head()

        # TorchMetrics (no storing full preds/labels)
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_pearson = torchmetrics.PearsonCorrCoef()
        self.val_spearman = torchmetrics.SpearmanCorrCoef()
        self.val_kendall = torchmetrics.KendallRankCorrCoef()

        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_pearson = torchmetrics.PearsonCorrCoef()
        self.test_spearman = torchmetrics.SpearmanCorrCoef()
        self.test_kendall = torchmetrics.KendallRankCorrCoef()

        # optional exports
        self.test_metrics: Dict[str, float] = {}
        self.test_predictions: List[Dict[str, float]] = []

    # ---------------- freezing ----------------

    def _freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_head(self):
        unfroze = False
        for attr in _HEAD_ATTRS:
            mod = getattr(self.model, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
                unfroze = True

        if not unfroze:
            for name, p in self.model.named_parameters():
                if any(
                    k in name.lower()
                    for k in ("classifier", "score", "head", "regression")
                ):
                    p.requires_grad = True

        if not any(p.requires_grad for p in self.model.parameters()):
            for _, p in list(self.model.named_parameters())[-200:]:
                p.requires_grad = True
                break

    # ---------------- forward ----------------

    def forward(self, **batch: Dict[str, Any]) -> torch.Tensor:
        out = self.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids", None),
            output_hidden_states=False,
        )
        return out.logits.squeeze(-1)

    # ---------------- steps ----------------

    def training_step(self, batch, batch_idx):
        preds = self(**batch)
        labels = batch["labels"].to(torch.float32).view_as(preds)
        loss = F.mse_loss(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(**batch)
        labels = batch["labels"].to(torch.float32).view_as(preds)

        loss = F.mse_loss(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.val_mse.update(preds, labels)
        self.val_pearson.update(preds, labels)
        self.val_spearman.update(preds, labels)
        self.val_kendall.update(preds, labels)

    def on_validation_epoch_end(self):
        mse = self.val_mse.compute()
        rmse = torch.sqrt(mse)
        pearson = self.val_pearson.compute()
        spearman = self.val_spearman.compute()
        kendall = self.val_kendall.compute()

        self.log("val_mse", mse, prog_bar=True)
        self.log("val_rmse", rmse)
        self.log("val_pearson", pearson)
        self.log("val_spearman", spearman)
        self.log("val_kendall", kendall)

        self.val_mse.reset()
        self.val_pearson.reset()
        self.val_spearman.reset()
        self.val_kendall.reset()

    def test_step(self, batch, batch_idx):
        preds = self(**batch)
        labels = batch["labels"].to(torch.float32).view_as(preds)

        self.test_mse.update(preds, labels)
        self.test_pearson.update(preds, labels)
        self.test_spearman.update(preds, labels)
        self.test_kendall.update(preds, labels)

        for p, l in zip(preds.detach().cpu(), labels.detach().cpu()):
            self.test_predictions.append(
                {"pred": float(p.item()), "label": float(l.item())}
            )

    def on_test_epoch_end(self):
        mse = self.test_mse.compute()
        rmse = torch.sqrt(mse)
        pearson = self.test_pearson.compute()
        spearman = self.test_spearman.compute()
        kendall = self.test_kendall.compute()

        self.log("test_mse", mse)
        self.log("test_rmse", rmse)
        self.log("test_pearson", pearson)
        self.log("test_spearman", spearman)
        self.log("test_kendall", kendall)

        self.test_metrics = {
            "mse": float(mse.item()),
            "rmse": float(rmse.item()),
            "pearson": float(pearson.item()),
            "spearman": float(spearman.item()),
            "kendall": float(kendall.item()),
        }

        self.test_mse.reset()
        self.test_pearson.reset()
        self.test_spearman.reset()
        self.test_kendall.reset()

    # ---------------- optim ----------------

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
