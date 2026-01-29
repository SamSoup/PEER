# /baselines/train_seq_cls.py

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
import yaml

from baselines.SeqCls import SequenceClassificationRegressor
from data.factory import build_pairwise_datamodule
from peer.utils import (
    ensure_hf_cache,
    set_seed,
    _maybe_enable_tensor_cores,
    ensure_hf_cache,
    summarize_trainable,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HF sequence classification regressor"
    )
    p.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Data / tokenization
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model used for BOTH tokenizer and model",
    )
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--combine_fields", action="store_true")
    p.add_argument("--combine_separator_token", type=str, default=None)

    # HF cache
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HF cache_dir override (defaults to HF_HOME or scratch)",
    )

    # Model / opt
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs/seqcls")

    # Trainer
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--monitor_metric", type=str, default="val_mse")
    p.add_argument("--monitor_mode", type=str, default="min")
    p.add_argument("--monitor_min_delta", type=float, default=0.01)
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

    merged = {
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
            "combine_fields": _cfg_get(
                cfg, "data.combine_fields", args.combine_fields
            ),
            "combine_separator_token": _cfg_get(
                cfg,
                "data.combine_separator_token",
                args.combine_separator_token,
            ),
        },
        "model": {
            "model_name": model_name,
            "cache_dir": cache_dir or ensure_hf_cache(),
            "lr": _cfg_get(cfg, "model.lr", args.lr),
            "weight_decay": _cfg_get(
                cfg, "model.weight_decay", args.weight_decay
            ),
            "freeze_encoder": _cfg_get(
                cfg, "model.freeze_encoder", args.freeze_encoder
            ),
            "output_dir": _cfg_get(cfg, "model.output_dir", args.output_dir),
        },
        "trainer": {
            "max_epochs": _cfg_get(cfg, "trainer.max_epochs", args.max_epochs),
            "patience": _cfg_get(cfg, "trainer.patience", args.patience),
            "precision": _cfg_get(cfg, "trainer.precision", args.precision),
            "monitor_metric": _cfg_get(
                cfg, "trainer.monitor_metric", args.monitor_metric
            ),
            "monitor_mode": _cfg_get(
                cfg, "trainer.monitor_mode", args.monitor_mode
            ),
            "monitor_min_delta": _cfg_get(
                cfg, "trainer.monitor_min_delta", args.monitor_min_delta
            ),
            "seed": _cfg_get(cfg, "trainer.seed", args.seed),
        },
    }
    return merged


def build_datamodule(cfg: Dict[str, Any]):
    d = cfg["data"]
    dm = build_pairwise_datamodule(
        dataset_name=d["dataset"],
        model_name=d["model_name"],
        max_seq_length=d["max_seq_length"],
        batch_size=d["train_batch_size"],
        tokenize_inputs=True,
        combine_fields=d["combine_fields"],
        combine_separator_token=d["combine_separator_token"],
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset name: {d['dataset']}")
    dm.train_batch_size = d["train_batch_size"]
    dm.eval_batch_size = d["eval_batch_size"]
    return dm


def _maybe_resize_token_embeddings(
    model: SequenceClassificationRegressor, dm
) -> None:
    tok = getattr(dm, "tokenizer", None)
    if tok is None:
        return

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    base = model.model
    if base.config.pad_token_id is None and tok.pad_token_id is not None:
        base.config.pad_token_id = tok.pad_token_id
        if getattr(base, "generation_config", None) is not None:
            base.generation_config.pad_token_id = tok.pad_token_id

    emb = base.get_input_embeddings()
    if emb is not None and emb.num_embeddings != len(tok):
        base.resize_token_embeddings(len(tok))


def build_model(cfg: Dict[str, Any], dm) -> SequenceClassificationRegressor:
    m = cfg["model"]
    # tokenization model and model must match
    if cfg["data"]["model_name"] != m["model_name"]:
        raise ValueError(
            "Tokenizer model and trained model must match. "
            f"data.model_name={cfg['data']['model_name']} vs model.model_name={m['model_name']}"
        )

    model = SequenceClassificationRegressor(
        model_name=m["model_name"],
        cache_dir=m["cache_dir"],
        lr=m["lr"],
        weight_decay=m["weight_decay"],
        freeze_encoder=m["freeze_encoder"],
    )
    _maybe_resize_token_embeddings(model, dm)
    return model


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
    model = build_model(cfg, dm)
    print(summarize_trainable(model, top_k=30))

    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="seqcls")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        monitor=cfg["trainer"]["monitor_metric"],
        mode=cfg["trainer"]["monitor_mode"],
        save_top_k=1,
        filename="best-{epoch:02d}-{val_mse:.4f}",
    )

    es_cb = EarlyStopping(
        monitor=cfg["trainer"]["monitor_metric"],
        mode=cfg["trainer"]["monitor_mode"],
        min_delta=cfg["trainer"]["monitor_min_delta"],
        patience=cfg["trainer"]["patience"],
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
