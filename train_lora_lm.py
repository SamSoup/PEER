"""
Lightning + PEFT (LoRA) trainer for causal LMs on prompt-masked cross-entropy.

Prompts are built from PROMPT_TEMPLATES; only the assistant answer tokens are
used for the loss (prompt tokens masked with -100).
"""

import os

os.environ["HF_HUB_DISABLE_MMAP"] = "1"  # disables safetensors mmap
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


import argparse
import json

from pathlib import Path
from typing import Any, Dict
import gc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
import yaml
from pytorch_lightning.strategies import SingleDeviceStrategy

from data import get_datamodule
from models.lora_lm import LoraCausalLMFinetuner
from peer.utils import ensure_hf_cache, set_seed
from peft import set_peft_model_state_dict
import psutil
import pathlib, threading, time


class MemTrace(pl.Callback):
    def _log(self, tag):
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        print(f"[{tag}] RSS={rss:.2f} GiB", flush=True)
        if torch.cuda.is_available():
            print(
                f"[{tag}] CUDA alloc={torch.cuda.memory_allocated()/1e9:.2f} GB "
                f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB",
                flush=True,
            )

    def setup(self, trainer, pl_module, stage=None):
        self._log(f"cb_setup_{stage}")

    def on_fit_start(self, trainer, pl_module):
        self._log("on_fit_start")

    def on_sanity_check_start(self, trainer, pl_module):
        self._log("sanity_start")

    def on_sanity_check_end(self, trainer, pl_module):
        self._log("sanity_end")

    def on_train_start(self, trainer, pl_module):
        self._log("on_train_start")

    def on_train_epoch_start(self, trainer, pl_module):
        self._log("epoch0_start")


def _maybe_enable_tensor_cores():
    if not torch.cuda.is_available():
        return
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA causal LM scorer")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--dataset", help="Dataset name for get_datamodule")
    parser.add_argument(
        "--model_name", help="HF model name (tokenizer + base model)"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--prompt_template", default="stsb")
    parser.add_argument("--use_cot", action="store_true")

    # Model / LoRA
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="*", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        default="bf16",
        help='Torch dtype for model weights: "auto", "bf16", "fp16", or "fp32".',
    )
    parser.add_argument("--output_dir", default="outputs/lora_lm")
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Passed to from_pretrained (e.g., "auto", "cuda").',
    )

    # Trainer
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--monitor_metric", default="val/loss")
    parser.add_argument("--monitor_mode", default="min")
    parser.add_argument("--monitor_min_delta", type=float, default=0.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    def _get(section: str, key: str, default: Any):
        if (
            section in cfg
            and key in cfg[section]
            and cfg[section][key] is not None
        ):
            return cfg[section][key]
        arg_val = getattr(args, key, None)
        return default if arg_val is None else arg_val

    merged = {
        "data": {
            "dataset": _get("data", "dataset", args.dataset),
            "model_name": _get("data", "model_name", args.model_name),
            "max_seq_length": _get(
                "data", "max_seq_length", args.max_seq_length
            ),
            "train_batch_size": _get(
                "data", "train_batch_size", args.train_batch_size
            ),
            "eval_batch_size": _get(
                "data", "eval_batch_size", args.eval_batch_size
            ),
            "prompt_template": _get(
                "data", "prompt_template", args.prompt_template
            ),
            "use_cot": _get("data", "use_cot", args.use_cot),
        },
        "model": {
            "model_name": _get("model", "model_name", args.model_name),
            "lr": _get("model", "lr", args.lr),
            "weight_decay": _get("model", "weight_decay", args.weight_decay),
            "lora_r": _get("model", "lora_r", args.lora_r),
            "lora_alpha": _get("model", "lora_alpha", args.lora_alpha),
            "lora_dropout": _get("model", "lora_dropout", args.lora_dropout),
            "lora_target_modules": _get(
                "model", "lora_target_modules", args.lora_target_modules
            ),
            "gradient_checkpointing": _get(
                "model", "gradient_checkpointing", args.gradient_checkpointing
            ),
            "output_dir": _get("model", "output_dir", args.output_dir),
            "torch_dtype": _get("model", "torch_dtype", args.torch_dtype),
            "device_map": _get("model", "device_map", args.device_map),
        },
        "trainer": {
            "max_epochs": _get("trainer", "max_epochs", args.max_epochs),
            "patience": _get("trainer", "patience", args.patience),
            "precision": _get("trainer", "precision", args.precision),
            "monitor_metric": _get(
                "trainer", "monitor_metric", args.monitor_metric
            ),
            "monitor_mode": _get("trainer", "monitor_mode", args.monitor_mode),
            "monitor_min_delta": _get(
                "trainer", "monitor_min_delta", args.monitor_min_delta
            ),
            "log_every_n_steps": _get(
                "trainer", "log_every_n_steps", args.log_every_n_steps
            ),
            "seed": _get("trainer", "seed", args.seed),
            "accumulate_grad_batches": _get(
                "trainer",
                "accumulate_grad_batches",
                args.accumulate_grad_batches,
            ),
        },
    }

    required = [
        ("data", "dataset"),
        ("data", "model_name"),
        ("model", "model_name"),
    ]
    missing = [
        f"{section}.{key}"
        for section, key in required
        if merged.get(section, {}).get(key) is None
    ]
    if missing:
        raise ValueError(
            f"Missing required config values: {', '.join(missing)}"
        )
    return merged


def build_datamodule(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    dm = get_datamodule(
        dataset_name=data_cfg["dataset"],
        model_name=data_cfg["model_name"],
        max_seq_length=data_cfg["max_seq_length"],
        batch_size=data_cfg["train_batch_size"],
        tokenize_inputs=True,
        prompt_template=data_cfg["prompt_template"],
        use_cot=data_cfg["use_cot"],
        lm=True,
    )
    dm.train_batch_size = data_cfg["train_batch_size"]
    dm.eval_batch_size = data_cfg["eval_batch_size"]
    return dm


def build_model(
    cfg: Dict[str, Any], dataset_key: str | None = None
) -> LoraCausalLMFinetuner:
    model_cfg = cfg["model"]
    return LoraCausalLMFinetuner(
        model_name=model_cfg["model_name"],
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        lora_r=model_cfg["lora_r"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        lora_target_modules=model_cfg["lora_target_modules"],
        gradient_checkpointing=model_cfg["gradient_checkpointing"],
        dataset_key=dataset_key or cfg["data"]["dataset"],
        device_map=model_cfg["device_map"],
        torch_dtype=model_cfg["torch_dtype"],
    )


def _read(path: str) -> str:
    try:
        return pathlib.Path(path).read_text().strip()
    except Exception:
        return "NA"


def _read_int(path: str) -> int:
    try:
        return int(_read(path))
    except Exception:
        return -1


def _gib(x: int) -> str:
    return "NA" if x < 0 else f"{x/1024**3:.2f} GiB"


def start_cgroup_monitor(interval=0.25):
    def loop():
        mem_max = _read("/sys/fs/cgroup/memory.max")
        while True:
            cur = _read_int("/sys/fs/cgroup/memory.current")
            stat = _read("/sys/fs/cgroup/memory.stat").splitlines()
            kv = {}
            for line in stat:
                k, v = line.split()
                kv[k] = int(v)
            anon = kv.get("anon", -1)
            file = kv.get("file", -1)
            shmem = kv.get("shmem", -1)
            print(
                f"[cgroup] current={_gib(cur)} max={mem_max} "
                f"anon={_gib(anon)} file={_gib(file)} shmem={_gib(shmem)}",
                flush=True,
            )
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


def train():
    ensure_hf_cache()
    _maybe_enable_tensor_cores()
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg["trainer"]["seed"])

    run_dir = Path(cfg["model"]["output_dir"])
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "adapters").mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))

    start_cgroup_monitor()

    dm = build_datamodule(cfg)

    dataset_key = getattr(dm, "dataset_key", cfg["data"]["dataset"])
    model = build_model(cfg, dataset_key=dataset_key)
    print(
        "RSS after model init:",
        psutil.Process(os.getpid()).memory_info().rss / (1024**3),
        "GiB",
        flush=True,
    )
    model = model.to("cuda:0")
    torch.cuda.synchronize()
    print(
        "RSS after model.to(cuda):",
        psutil.Process(os.getpid()).memory_info().rss / (1024**3),
        "GiB",
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()

    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="lora_lm")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        monitor=cfg["trainer"]["monitor_metric"],
        mode=cfg["trainer"]["monitor_mode"],
        save_top_k=1,
        filename="lora-{epoch:02d}-{val_loss:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
    )
    es_cb = EarlyStopping(
        monitor=cfg["trainer"]["monitor_metric"],
        mode=cfg["trainer"]["monitor_mode"],
        min_delta=cfg["trainer"]["monitor_min_delta"],
        patience=cfg["trainer"]["patience"],
    )

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available =", torch.cuda.is_available())
    print("torch.cuda.device_count =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current_device =", torch.cuda.current_device())
        print("device_name =", torch.cuda.get_device_name(0))

    print("Got here!")
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=[0],
        strategy=SingleDeviceStrategy(device=torch.device("cuda", 0)),
        max_epochs=cfg["trainer"]["max_epochs"],
        # precision=cfg["trainer"]["precision"],
        precision="bf16-true",
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        callbacks=[es_cb, MemTrace()],
        logger=logger,
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        num_sanity_val_steps=0,  # remove pre-epoch extras
        enable_checkpointing=False,  # temporarily
        enable_model_summary=False,
    )
    print("Got here after trainer!")
    print("first param device:", next(model.parameters()).device)
    print(
        "RSS before fit:",
        psutil.Process(os.getpid()).memory_info().rss / (1024**3),
        "GiB",
        flush=True,
    )

    trainer.fit(model, datamodule=dm)
    print("Got here after trainer fit!")
    test_metrics = trainer.test(model, datamodule=dm)

    metrics_path = run_dir / "test_metrics.json"
    payload: Dict[str, Any] = test_metrics[0] if test_metrics else {}
    if getattr(model, "test_metrics", None):
        payload.update(model.test_metrics)
    if payload:
        payload["hyperparams"] = {
            "lora_r": cfg["model"]["lora_r"],
            "lora_alpha": cfg["model"]["lora_alpha"],
            "lora_dropout": cfg["model"]["lora_dropout"],
            "lora_target_modules": cfg["model"]["lora_target_modules"],
            "lr": cfg["model"]["lr"],
            "seed": cfg["trainer"]["seed"],
        }
        metrics_path.write_text(json.dumps(payload, indent=2, default=str))
        print("Saved test metrics:", metrics_path)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        adapter_dir = run_dir / "adapters" / "best"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        # Load adapter weights from checkpoint into current model to avoid double-loading base weights
        ckpt = torch.load(best_ckpt, map_location="cpu")
        adapter_sd = {
            k[len("model.") :]: v
            for k, v in ckpt.get("state_dict", {}).items()
            if k.startswith("model.")
        }
        if adapter_sd:
            set_peft_model_state_dict(model.model, adapter_sd)
        model.model.save_pretrained(str(adapter_dir))
        model.tokenizer.save_pretrained(str(adapter_dir))
        print("Saved best LoRA adapter to:", adapter_dir)
    else:
        adapter_dir = run_dir / "adapters" / "last"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(str(adapter_dir))
        model.tokenizer.save_pretrained(str(adapter_dir))
        print("Saved last LoRA adapter to:", adapter_dir)

    preds = getattr(model, "test_predictions", None)
    if preds:
        preds_path = run_dir / "test_predictions.jsonl"
        with open(preds_path, "w") as f:
            for record in preds:
                f.write(json.dumps(record) + "\n")
        print("Saved test predictions:", preds_path)


if __name__ == "__main__":
    train()
