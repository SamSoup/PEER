# /baselines/train_lora_lm.py

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
import torch
import yaml

from baselines.Lora import LoraCausalLMRegressor
from data.factory import build_prompt_masked_lm_datamodule
from peer.utils import (
    ensure_hf_cache,
    set_seed,
    _maybe_enable_tensor_cores,
    summarize_trainable,
)


def _default_cache_dir() -> str:
    return os.environ.get("HF_HOME") or "/scratch/06782/ysu707/.cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA/QLoRA finetune CausalLM on prompt-masked regression labels"
    )
    p.add_argument("--config", type=str, default=None)

    # data
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--combine_fields", action="store_true")
    p.add_argument("--combine_separator_token", type=str, default=None)
    p.add_argument("--add_eos", action="store_true")
    p.add_argument("--no_chat_template", action="store_true")

    # hf cache
    p.add_argument("--cache_dir", type=str, default=None)

    # optim
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--target_modules", type=str, default=None)

    # quantization
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    p.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    p.add_argument("--no_bnb_4bit_use_double_quant", action="store_true")
    p.add_argument(
        "--bnb_4bit_compute_dtype", type=str, default="bf16"
    )  # bf16|fp16|fp32
    p.add_argument("--llm_int8_threshold", type=float, default=6.0)

    # attention
    p.add_argument(
        "--attn_implementation", type=str, default="flash_attention_2"
    )  # flash_attention_2|sdpa|eager

    # generation eval (test)
    p.add_argument("--gen_max_new_tokens", type=int, default=8)
    p.add_argument("--gen_temperature", type=float, default=0.0)
    p.add_argument("--gen_top_p", type=float, default=1.0)

    # trainer
    p.add_argument("--output_dir", type=str, default="outputs/lora_lm")
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any):
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


def _parse_target_modules(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    out = [p for p in parts if p]
    return out or None


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    dataset = _cfg_get(cfg, "data.dataset", args.dataset)
    model_name = _cfg_get(cfg, "model.model_name", args.model_name) or _cfg_get(
        cfg, "data.model_name", args.model_name
    )
    cache_dir = _cfg_get(cfg, "model.cache_dir", args.cache_dir) or _cfg_get(
        cfg, "data.cache_dir", args.cache_dir
    )

    if dataset is None or model_name is None:
        raise ValueError(
            "Missing required values: data.dataset and model.model_name (or --dataset/--model_name)."
        )

    # double-quant flag handling
    dq_default = True
    dq = _cfg_get(cfg, "model.bnb_4bit_use_double_quant", None)
    if dq is None:
        dq = (
            args.bnb_4bit_use_double_quant
            if not args.no_bnb_4bit_use_double_quant
            else False
        )
    dq = bool(dq)

    return {
        "data": {
            "dataset": dataset,
            "model_name": model_name,
            "max_seq_length": _cfg_get(
                cfg, "data.max_seq_length", args.max_seq_length
            ),
            "train_batch_size": _cfg_get(
                cfg, "data.train_batch_size", args.train_batch_size
            ),
            "eval_batch_size": _cfg_get(
                cfg, "data.eval_batch_size", args.eval_batch_size
            ),
            "accumulate_grad_batches": _cfg_get(
                cfg,
                "data.accumulate_grad_batches",
                args.accumulate_grad_batches,
            ),
            "combine_fields": _cfg_get(
                cfg, "data.combine_fields", args.combine_fields
            ),
            "combine_separator_token": _cfg_get(
                cfg,
                "data.combine_separator_token",
                args.combine_separator_token,
            ),
            "add_eos": _cfg_get(cfg, "data.add_eos", args.add_eos),
            "use_chat_template": not _cfg_get(
                cfg, "data.no_chat_template", args.no_chat_template
            ),
        },
        "model": {
            "model_name": model_name,
            "cache_dir": cache_dir or _default_cache_dir(),
            "lr": _cfg_get(cfg, "model.lr", args.lr),
            "weight_decay": _cfg_get(
                cfg, "model.weight_decay", args.weight_decay
            ),
            "output_dir": _cfg_get(cfg, "model.output_dir", args.output_dir),
            "lora_r": _cfg_get(cfg, "model.lora_r", args.lora_r),
            "lora_alpha": _cfg_get(cfg, "model.lora_alpha", args.lora_alpha),
            "lora_dropout": _cfg_get(
                cfg, "model.lora_dropout", args.lora_dropout
            ),
            "lora_bias": _cfg_get(cfg, "model.lora_bias", args.lora_bias),
            "target_modules": _cfg_get(
                cfg, "model.target_modules", args.target_modules
            ),
            "load_in_4bit": _cfg_get(
                cfg, "model.load_in_4bit", args.load_in_4bit
            ),
            "load_in_8bit": _cfg_get(
                cfg, "model.load_in_8bit", args.load_in_8bit
            ),
            "bnb_4bit_quant_type": _cfg_get(
                cfg, "model.bnb_4bit_quant_type", args.bnb_4bit_quant_type
            ),
            "bnb_4bit_use_double_quant": dq if dq is not None else dq_default,
            "bnb_4bit_compute_dtype": _cfg_get(
                cfg, "model.bnb_4bit_compute_dtype", args.bnb_4bit_compute_dtype
            ),
            "llm_int8_threshold": _cfg_get(
                cfg, "model.llm_int8_threshold", args.llm_int8_threshold
            ),
            "attn_implementation": _cfg_get(
                cfg, "model.attn_implementation", args.attn_implementation
            ),
            "gen_max_new_tokens": _cfg_get(
                cfg, "model.gen_max_new_tokens", args.gen_max_new_tokens
            ),
            "gen_temperature": _cfg_get(
                cfg, "model.gen_temperature", args.gen_temperature
            ),
            "gen_top_p": _cfg_get(cfg, "model.gen_top_p", args.gen_top_p),
        },
        "trainer": {
            "max_epochs": _cfg_get(cfg, "trainer.max_epochs", args.max_epochs),
            "patience": _cfg_get(cfg, "trainer.patience", args.patience),
            "precision": _cfg_get(cfg, "trainer.precision", args.precision),
            "seed": _cfg_get(cfg, "trainer.seed", args.seed),
        },
    }


def build_datamodule(cfg: Dict[str, Any]):
    d = cfg["data"]
    dm = build_prompt_masked_lm_datamodule(
        dataset_name=d["dataset"],
        model_name=d["model_name"],
        max_seq_length=d["max_seq_length"],
        train_batch_size=d["train_batch_size"],
        eval_batch_size=d["eval_batch_size"],
        combine_fields=d["combine_fields"],
        combine_separator_token=d["combine_separator_token"],
        add_eos=d["add_eos"],
        use_chat_template_if_available=d["use_chat_template"],
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset name: {d['dataset']}")
    return dm


def build_model(cfg: Dict[str, Any]) -> LoraCausalLMRegressor:
    m = cfg["model"]
    d = cfg["data"]
    return LoraCausalLMRegressor(
        model_name=m["model_name"],
        dataset_key=d["dataset"],
        cache_dir=m["cache_dir"],
        lr=m["lr"],
        weight_decay=m["weight_decay"],
        lora_r=m["lora_r"],
        lora_alpha=m["lora_alpha"],
        lora_dropout=m["lora_dropout"],
        lora_bias=m["lora_bias"],
        target_modules=_parse_target_modules(m["target_modules"]),
        load_in_4bit=bool(m["load_in_4bit"]),
        load_in_8bit=bool(m["load_in_8bit"]),
        bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(m["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=m["bnb_4bit_compute_dtype"],
        llm_int8_threshold=float(m["llm_int8_threshold"]),
        attn_implementation=m["attn_implementation"],
        gen_max_new_tokens=m["gen_max_new_tokens"],
        gen_temperature=m["gen_temperature"],
        gen_top_p=m["gen_top_p"],
    )


def train():
    ensure_hf_cache()
    _maybe_enable_tensor_cores()

    args = parse_args()
    cfg = load_config(args)

    set_seed(cfg["trainer"]["seed"])
    pl.seed_everything(cfg["trainer"]["seed"], workers=True)

    run_dir = Path(cfg["model"]["output_dir"])
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "adapters").mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))

    dm = build_datamodule(cfg)
    model = build_model(cfg)

    # attach tokenizer for generation + gold decode
    model.tokenizer = dm.tokenizer

    print(summarize_trainable(model, top_k=30))

    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="lora_lm")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_weights_only=True,  # adapter-only via LightningModule.state_dict override
    )
    es_cb = EarlyStopping(
        monitor="val_loss", mode="min", patience=cfg["trainer"]["patience"]
    )

    # Fix 3: Explicit hardware detection for Stampede3/idev
    gpu_count = torch.cuda.device_count()

    # Use DDP if multiple GPUs found; avoid 'auto' which triggers SLURM checks
    strategy = "auto"
    if gpu_count > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
        print(f"Detected {gpu_count} GPUs. Using DDP Strategy.")

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        precision=cfg["trainer"]["precision"],
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        log_every_n_steps=10,
        accelerator="gpu",
        devices=gpu_count if gpu_count > 0 else 1,
        strategy=strategy,
        accumulate_grad_batches=cfg["data"]["accumulate_grad_batches"],
    )

    trainer.fit(model, datamodule=dm)
    # NOTE: To save memory, we only test on the current module in memory
    test_metrics = trainer.test(model, datamodule=dm)

    if trainer.global_rank == 0:
        adapter_dir = run_dir / "adapters" / "best"
        model.save_adapters(str(adapter_dir))

        payload: Dict[str, Any] = (
            test_metrics[0] if test_metrics else {}
        ).copy()
        if getattr(model, "test_metrics", None):
            payload.update(model.test_metrics)

        if payload:
            path = run_dir / "test_metrics.json"
            path.write_text(json.dumps(payload, indent=2, default=str))
            print("Saved test metrics:", path)

        preds = getattr(model, "test_predictions", None)
        if preds:
            path = run_dir / "test_predictions.jsonl"
            with open(path, "w") as f:
                for r in preds:
                    f.write(json.dumps(r) + "\n")
            print("Saved test predictions:", path)

        print("Saved adapters:", adapter_dir)


if __name__ == "__main__":
    train()
