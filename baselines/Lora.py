# /baselines/Lora.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from data.pair_dataset_registry import parse_score

try:
    from peft import LoraConfig, get_peft_model

    # adapter-only checkpointing
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    # QLoRA prep
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None

except Exception as e:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None
    prepare_model_for_kbit_training = None
    _PEFT_IMPORT_ERROR = e
else:
    _PEFT_IMPORT_ERROR = None


def _infer_target_modules(model) -> List[str]:
    mt = getattr(getattr(model, "config", None), "model_type", "") or ""
    mt = mt.lower()
    if any(k in mt for k in ("llama", "mistral", "qwen", "gemma")):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if "gpt2" in mt:
        return ["c_attn", "c_proj"]
    return ["q_proj", "v_proj"]


def _dtype_from_str(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype string: {name} (use bf16|fp16|fp32)")


def _ensure_bitsandbytes_available():
    try:
        import bitsandbytes as _  # noqa: F401
    except Exception as e:
        raise ImportError(
            "You requested 4-bit/8-bit loading but bitsandbytes is not available. "
            "Install bitsandbytes compatible with your CUDA environment."
        ) from e


class LoraCausalLMRegressor(pl.LightningModule):
    """
    LoRA/QLoRA finetune a CausalLM on prompt-masked labels (labels=-100 on prompt tokens).

    - Training/val: HF LM loss (ignores -100).
    - Test: generate, parse numeric score, compare to decoded gold score.
    - Checkpointing: adapter-only via peft state_dict overrides.
    """

    def __init__(
        self,
        model_name: str,
        dataset_key: str,
        *,
        cache_dir: Optional[str] = None,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        # LoRA params
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        target_modules: Optional[List[str]] = None,
        # quantization
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str = "bf16",  # bf16|fp16|fp32
        llm_int8_threshold: float = 6.0,
        # attention
        attn_implementation: str = "flash_attention_2",  # flash_attention_2|sdpa|eager
        # generation (test)
        gen_max_new_tokens: int = 8,
        gen_temperature: float = 0.0,
        gen_top_p: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if get_peft_model is None:
            raise ImportError(
                "peft is required for LoRA baseline but could not be imported. "
                f"Original error: {_PEFT_IMPORT_ERROR}"
            )

        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose only one: load_in_4bit or load_in_8bit")

        if load_in_4bit or load_in_8bit:
            _ensure_bitsandbytes_available()

        self.dataset_key = dataset_key
        self.tokenizer = None  # attach from datamodule in training script

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

        quant_cfg = None
        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
                bnb_4bit_compute_dtype=_dtype_from_str(bnb_4bit_compute_dtype),
            )
        elif load_in_8bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=float(llm_int8_threshold),
            )

        # Try requested attn impl; fall back to sdpa then eager if needed.
        base = None
        last_err = None
        for attn_impl in [attn_implementation, "sdpa", "eager"]:
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    cache_dir=cache_dir,
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_impl,
                    quantization_config=quant_cfg,
                    # device_map=None,
                )
                break
            except Exception as e:
                last_err = e
                base = None
                continue
        if base is None:
            raise RuntimeError(
                f"Failed to load model with attn_implementation fallback. Last error: {last_err}"
            )

        # Pad token safety
        if (
            not hasattr(base.config, "pad_token_id")
            or base.config.pad_token_id is None
        ):
            # Gemma-3 uses eos_token_id which can be a list or int
            eos_id = getattr(base.config, "eos_token_id", None)
            if isinstance(eos_id, list):
                base.config.pad_token_id = int(eos_id[0])
            elif eos_id is not None:
                base.config.pad_token_id = int(eos_id)
            else:
                base.config.pad_token_id = 0

        # If k-bit base, prepare for QLoRA-style training (layer norms, gradients, etc.)
        if (
            load_in_4bit or load_in_8bit
        ) and prepare_model_for_kbit_training is not None:
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=True
            )

        if target_modules is None:
            target_modules = _infer_target_modules(base)

        lora_cfg = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias=str(lora_bias),
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(base, lora_cfg)

        # memory / training safety
        self.model.config.use_cache = False
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(self.model, "enable_input_require_grads"):
            try:
                self.model.enable_input_require_grads()
            except Exception:
                pass

        # Metrics (generation-based) for test
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_pearson = torchmetrics.PearsonCorrCoef()
        self.test_spearman = torchmetrics.SpearmanCorrCoef()
        self.test_kendall = torchmetrics.KendallRankCorrCoef()

        self.test_metrics: Dict[str, float] = {}
        self.test_predictions: List[Dict[str, Any]] = []

    # ---------------- checkpointing: save ONLY adapters ----------------

    def state_dict(self, *args, **kwargs):
        if get_peft_model_state_dict is None:
            return super().state_dict(*args, **kwargs)
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict, strict: bool = True):
        if set_peft_model_state_dict is None:
            return super().load_state_dict(state_dict, strict=strict)
        set_peft_model_state_dict(self.model, state_dict)
        return

    # ---------------- pad token safety ----------------
    def on_fit_start(self):
        if self.model.config.pad_token_id is None:
            if (
                self.tokenizer is not None
                and self.tokenizer.pad_token_id is not None
            ):
                self.model.config.pad_token_id = int(
                    self.tokenizer.pad_token_id
                )
            elif self.model.config.eos_token_id is not None:
                self.model.config.pad_token_id = int(
                    self.model.config.eos_token_id
                )
        else:
            self.model.config.pad_token_id = self._as_int_token_id(
                self.model.config.pad_token_id, default=0
            )
        # Ensure model and tokenizer agree on the PAD ID as a standard int
        if self.tokenizer is not None:
            self.model.config.pad_token_id = self._as_int_token_id(
                self.tokenizer.pad_token_id
            )
        print(
            f"Rank {self.global_rank} | Padding Token ID: {self.model.config.pad_token_id}"
        )

    # ---------------- forward/steps ----------------

    def forward(self, **batch: Dict[str, Any]):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch, use_cache=False)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch, use_cache=False)
        loss = out.loss
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        out = self.model(**batch, use_cache=False)
        self.log(
            "test_loss",
            out.loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        preds, pred_texts = self._generate_pred_scores(batch)
        labels, label_texts = self._decode_gold_scores(batch)

        mask = torch.isfinite(preds) & torch.isfinite(labels)
        if mask.any():
            self.test_mse.update(preds[mask], labels[mask])
            self.test_pearson.update(preds[mask], labels[mask])
            self.test_spearman.update(preds[mask], labels[mask])
            self.test_kendall.update(preds[mask], labels[mask])

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

        self.log("test_mse", mse, sync_dist=True)
        self.log("test_rmse", rmse, sync_dist=True)
        self.log("test_pearson", pearson, sync_dist=True)
        self.log("test_spearman", spearman, sync_dist=True)
        self.log("test_kendall", kendall, sync_dist=True)

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
        Build prompt-only inputs for generation WITHOUT introducing right-padding.

        We:
          1) find where completion starts (first label != -100),
          2) slice prompt tokens up to that point,
          3) strip any existing pads via attention_mask,
          4) re-pad with tokenizer.pad (respects tokenizer.padding_side, set to 'left').
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Attach tokenizer before testing: model.tokenizer = dm.tokenizer"
            )

        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: torch.Tensor = batch["attention_mask"]
        labels: torch.Tensor = batch["labels"]

        # Find completion start index per example (first non -100 label)
        prompt_lens: List[int] = []
        for i in range(labels.size(0)):
            idx = (labels[i] != -100).nonzero(as_tuple=False)
            prompt_lens.append(
                int(idx[0].item()) if idx.numel() else labels.size(1)
            )

        # Build per-example feature dicts with pads stripped
        feats: List[Dict[str, Any]] = []
        for i, L in enumerate(prompt_lens):
            ids = input_ids[i, :L]
            attn = attention_mask[i, :L]

            # Strip existing pads so tokenizer.pad can re-pad cleanly
            ids = ids[attn.bool()]
            feats.append(
                {
                    "input_ids": ids.detach().cpu().tolist(),
                    "attention_mask": [1] * int(ids.numel()),
                }
            )

        # Ensure left padding for decoder-only generation
        self.tokenizer.padding_side = "left"

        padded = self.tokenizer.pad(feats, padding=True, return_tensors="pt")
        prompt_ids = padded["input_ids"].to(self.device)
        prompt_attn = padded["attention_mask"].to(self.device)

        return prompt_ids, prompt_attn, prompt_lens

    @torch.no_grad()
    def _generate_pred_scores(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate score strings from prompt-only inputs, then parse them.

        Because we re-pad prompts to a common length, the generated "new tokens"
        start at input_len for all examples.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Attach tokenizer before testing: model.tokenizer = dm.tokenizer"
            )

        prompt_ids, prompt_attn, _prompt_lens = self._prompt_only_inputs(batch)
        input_len = int(prompt_ids.size(1))

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
            # Everything after input_len is newly generated
            tail = gen[i, input_len:].detach().cpu().tolist()
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

    def _as_int_token_id(self, x, default: int = 0) -> int:
        if x is None:
            return default
        if isinstance(x, (list, tuple)):
            return int(x[0]) if len(x) > 0 else default
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)

    # ---------------- optim ----------------

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    # ---------------- saving adapters ----------------

    def save_adapters(self, out_dir: str):
        self.model.save_pretrained(out_dir)
