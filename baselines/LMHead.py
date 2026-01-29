# /baselines/LMHead.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoConfig, AutoModelForCausalLM

from data.pair_dataset_registry import parse_score


_LM_HEAD_ATTRS = (
    "lm_head",
    "embed_out",
    "output_projection",
    "output_layer",
    "language_model_head",
)


class CausalLMHeadRegressor(pl.LightningModule):
    """
    Trains only the CausalLM head on prompt-masked labels (labels=-100 on prompt tokens).

    Testing:
      - Generate from prompt-only prefix (not including the gold completion).
      - Parse numeric score from generated text via registry parse_score(dataset_key).
      - Compute MSE/RMSE/Pearson/Spearman/Kendall vs gold decoded from labels.
      - Save per-example predictions for debugging.

    Notes:
      - During val: we keep it cheap and log only val_loss (no generation).
      - During test: we do generation + metrics.
    """

    def __init__(
        self,
        model_name: str,
        dataset_key: str,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        freeze_base: bool = True,
        cache_dir: Optional[str] = None,
        # generation config (used in test)
        gen_max_new_tokens: int = 8,
        gen_temperature: float = 0.0,  # 0.0 => greedy
        gen_top_p: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_key = dataset_key
        self.tokenizer = (
            None  # attach in train script: model.tokenizer = dm.tokenizer
        )

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
        )

        if freeze_base:
            for p in self.model.parameters():
                p.requires_grad = False
            self._unfreeze_lm_head()

        # Test metrics (generation-based)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_pearson = torchmetrics.PearsonCorrCoef()
        self.test_spearman = torchmetrics.SpearmanCorrCoef()
        self.test_kendall = torchmetrics.KendallRankCorrCoef()

        self.test_metrics: Dict[str, float] = {}
        self.test_predictions: List[Dict[str, Any]] = []

    # ---------------- freezing ----------------

    def _unfreeze_lm_head(self):
        for attr in _LM_HEAD_ATTRS:
            head = getattr(self.model, attr, None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True
                return

        # fallback by name
        for name, p in self.model.named_parameters():
            lname = name.lower()
            if any(
                k in lname
                for k in (
                    "lm_head",
                    "output",
                    "embed_out",
                    "projection",
                    "language_model_head",
                )
            ):
                p.requires_grad = True

        if not any(p.requires_grad for p in self.model.parameters()):
            # last-resort: unfreeze a small tensor
            for _, p in list(self.model.named_parameters())[-200:]:
                p.requires_grad = True
                break

    # ---------------- pad token safety ----------------

    def on_fit_start(self):
        # Ensure pad_token_id is set for generation/padding. Llama often needs this.
        if self.model.config.pad_token_id is None:
            if (
                self.tokenizer is not None
                and self.tokenizer.pad_token_id is not None
            ):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            elif self.model.config.eos_token_id is not None:
                self.model.config.pad_token_id = self.model.config.eos_token_id

    # ---------------- forward/steps ----------------

    def forward(self, **batch: Dict[str, Any]):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        # still log LM loss
        out = self.model(**batch)
        self.log(
            "test_loss", out.loss, prog_bar=True, on_epoch=True, on_step=False
        )

        preds, pred_texts = self._generate_pred_scores(batch)
        labels, label_texts = self._decode_gold_scores(batch)

        mask = torch.isfinite(preds) & torch.isfinite(labels)
        if mask.any():
            self.test_mse.update(preds[mask], labels[mask])
            self.test_pearson.update(preds[mask], labels[mask])
            self.test_spearman.update(preds[mask], labels[mask])
            self.test_kendall.update(preds[mask], labels[mask])

        # Save per-example
        for p, y, gen_txt, gold_txt, ok in zip(
            preds.detach().cpu().tolist(),
            labels.detach().cpu().tolist(),
            pred_texts,
            label_texts,
            mask.detach().cpu().tolist(),
        ):
            self.test_predictions.append(
                {
                    "pred": float(p) if p == p else None,
                    "label": float(y) if y == y else None,
                    "gen": gen_txt,
                    "gold_text": gold_txt,
                    "ok": bool(ok),
                }
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

    # ---------------- generation helpers ----------------

    def _prompt_only_inputs(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Returns prompt-only (input_ids, attention_mask, prompt_lens) where prompt_lens[i]
        is the prefix length before completion starts (first label != -100).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        prompt_lens: List[int] = []
        for i in range(labels.size(0)):
            idx = (labels[i] != -100).nonzero(as_tuple=False)
            prompt_lens.append(
                int(idx[0].item()) if idx.numel() else labels.size(1)
            )

        max_prompt = max(prompt_lens) if prompt_lens else input_ids.size(1)

        prompt_ids = input_ids[:, :max_prompt].clone()
        prompt_attn = attention_mask[:, :max_prompt].clone()

        pad_id = self.model.config.pad_token_id
        if pad_id is None:
            pad_id = 0

        # pad variable prompt lengths
        for i, L in enumerate(prompt_lens):
            if L < max_prompt:
                prompt_ids[i, L:] = pad_id
                prompt_attn[i, L:] = 0

        return prompt_ids, prompt_attn, prompt_lens

    def _decode_gold_scores(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[str]]:
        if self.tokenizer is None:
            raise RuntimeError(
                "Attach tokenizer before testing: model.tokenizer = dm.tokenizer"
            )

        gold_vals: List[float] = []
        gold_texts: List[str] = []

        for lab in batch["labels"]:
            ids = lab[lab != -100].detach().cpu().tolist()
            txt = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            gold_texts.append(txt)
            try:
                y = parse_score(txt, self.dataset_key)
            except Exception:
                y = float("nan")
            gold_vals.append(float(y))

        return (
            torch.tensor(gold_vals, device=self.device, dtype=torch.float32),
            gold_texts,
        )

    @torch.no_grad()
    def _generate_pred_scores(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[str]]:
        if self.tokenizer is None:
            raise RuntimeError(
                "Attach tokenizer before testing: model.tokenizer = dm.tokenizer"
            )

        prompt_ids, prompt_attn, prompt_lens = self._prompt_only_inputs(batch)

        temperature = float(self.hparams.gen_temperature)
        top_p = float(self.hparams.gen_top_p)
        do_sample = temperature > 1e-8

        gen = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=int(self.hparams.gen_max_new_tokens),
            do_sample=do_sample,
            temperature=(temperature if do_sample else None),
            top_p=(top_p if do_sample else None),
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )

        pred_vals: List[float] = []
        pred_texts: List[str] = []

        for i in range(gen.size(0)):
            tail = gen[i, prompt_lens[i] :].detach().cpu().tolist()
            txt = self.tokenizer.decode(tail, skip_special_tokens=True).strip()
            pred_texts.append(txt)
            try:
                v = parse_score(txt, self.dataset_key)
            except Exception:
                v = float("nan")
            pred_vals.append(float(v))

        return (
            torch.tensor(pred_vals, device=self.device, dtype=torch.float32),
            pred_texts,
        )

    # ---------------- optim ----------------

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
