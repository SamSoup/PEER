# /baselines/train_lm_head.py

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import yaml

from baselines.LMHead import CausalLMHeadRegressor
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
        description="Train CausalLM head only with prompt-masked labels"
    )
    p.add_argument("--config", type=str, default=None)

    # data
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument(
        "--model_name", type=str, default=None
    )  # tokenizer + model must match
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--combine_fields", action="store_true")
    p.add_argument("--combine_separator_token", type=str, default=None)
    p.add_argument("--add_eos", action="store_true")
    p.add_argument("--no_chat_template", action="store_true")

    # hf cache
    p.add_argument("--cache_dir", type=str, default=None)

    # optim
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--freeze_base", action="store_true")

    # generation eval (test-time)
    p.add_argument("--gen_max_new_tokens", type=int, default=8)
    p.add_argument("--gen_temperature", type=float, default=0.0)  # 0 => greedy
    p.add_argument("--gen_top_p", type=float, default=1.0)

    # trainer
    p.add_argument("--output_dir", type=str, default="outputs/lm_head")
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

    return {
        "data": {
            "dataset": dataset,
            "model_name": model_name,
            "max_seq_length": _cfg_get(
                cfg, "data.max_seq_length", args.max_seq_length
            ),
            "batch_size": _cfg_get(cfg, "data.batch_size", args.batch_size),
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
            "freeze_base": _cfg_get(cfg, "model.freeze_base", args.freeze_base),
            "output_dir": _cfg_get(cfg, "model.output_dir", args.output_dir),
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
        batch_size=d["batch_size"],
        combine_fields=d["combine_fields"],
        combine_separator_token=d["combine_separator_token"],
        add_eos=d["add_eos"],
        use_chat_template_if_available=d["use_chat_template"],
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset name: {d['dataset']}")
    return dm


def build_model(cfg: Dict[str, Any]) -> CausalLMHeadRegressor:
    m = cfg["model"]
    d = cfg["data"]
    return CausalLMHeadRegressor(
        model_name=m["model_name"],
        dataset_key=d["dataset"],
        cache_dir=m["cache_dir"],
        lr=m["lr"],
        weight_decay=m["weight_decay"],
        freeze_base=m["freeze_base"],
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
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))

    dm = build_datamodule(cfg)
    model = build_model(cfg)

    # Attach tokenizer so LMHead can decode/generate/parse.
    model.tokenizer = dm.tokenizer
    model.tokenizer.padding_side = "left"

    print(summarize_trainable(model, top_k=30))

    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="lm_head")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    es_cb = EarlyStopping(
        monitor="val_loss", mode="min", patience=cfg["trainer"]["patience"]
    )

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        precision=cfg["trainer"]["precision"],
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(datamodule=dm, ckpt_path="best")

    payload: Dict[str, Any] = (test_metrics[0] if test_metrics else {}).copy()
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


if __name__ == "__main__":
    train()
