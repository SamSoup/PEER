import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as tmf


def _maybe_enable_tensor_cores():
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def huber_loss(y_hat, y, delta: float = 0.5):
    """
    Smooth L1 / Huber loss with configurable delta.
    Expects y_hat, y of shape [B] or [B, 1].
    """
    y_hat = y_hat.view(-1)
    y = y.view(-1)
    return F.huber_loss(y_hat, y, delta=delta)


def compute_label_stats(dataloader):
    """Compute mean/std of labels over a dataloader."""
    total = 0.0
    total_sq = 0.0
    n = 0
    for batch in dataloader:
        labels = batch["labels"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.float().view(-1)
        total += labels.sum().item()
        total_sq += (labels * labels).sum().item()
        n += labels.numel()
    mean = total / max(n, 1)
    var = total_sq / max(n, 1) - mean * mean
    var = max(var, 0.0)
    std = var**0.5
    return mean, std


def set_label_stats_from_loader(embedder, dataloader):
    mean, std = compute_label_stats(dataloader)
    embedder.set_stats(mean, std if std > 0 else 1.0)
    return mean, std


def regression_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute mse, rmse, pearson, spearman, kendall (where available).
    """
    preds = preds.view(-1)
    labels = labels.view_as(preds)
    mse = torch.mean((preds - labels) ** 2).item()
    rmse = float(torch.sqrt(torch.tensor(mse)))
    pearson = float(torch.corrcoef(torch.stack([preds, labels]))[0, 1])
    spearman = float(tmf.spearman_corrcoef(preds, labels))
    kendall = float(tmf.kendall_rank_corrcoef(preds, labels))

    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_hf_cache(default_cache: str = "/scratch/06782/ysu707/.cache"):
    """
    Set HF cache env vars to a writable location if not already provided.
    """
    return os.environ.get("HF_HOME") or default_cache


from typing import Iterable, Optional, Sequence, Tuple


def summarize_trainable(
    model,
    *,
    top_k: int = 50,
    name_contains: Optional[Sequence[str]] = None,
    print_all: bool = False,
) -> str:
    """
    Returns a human-readable summary of trainable vs total params + the largest trainable tensors.

    Args:
      model: nn.Module / LightningModule
      top_k: how many trainable tensors to list (largest first)
      name_contains: optional allowlist filter; only show params whose name contains any of these substrings
      print_all: if True, list all matching trainable tensors (ignores top_k)

    Usage:
      print(summarize_trainable(model))
      self.print(summarize_trainable(self.model))  # inside LightningModule
    """
    total = 0
    trainable = 0
    rows: list[Tuple[str, int, Tuple[int, ...]]] = []

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            rows.append((name, n, tuple(p.shape)))

    frac = (trainable / total) if total else 0.0

    # optional name filter (for display only)
    if name_contains:
        filt = tuple(name_contains)
        rows = [r for r in rows if any(s in r[0] for s in filt)]

    rows.sort(key=lambda x: x[1], reverse=True)
    shown = rows if print_all else rows[:top_k]

    lines = []
    lines.append(f"Trainable: {trainable:,} / {total:,} ({100*frac:.8f}%)")
    lines.append(f"Trainable tensors: {len(rows):,}")
    if name_contains:
        lines.append(f"Display filter: name_contains={list(name_contains)}")
    if not rows:
        lines.append("(no trainable parameters)")
        return "\n".join(lines)

    lines.append("Largest trainable tensors:")
    for name, n, shape in shown:
        lines.append(f"  {n:>12,}  {name}  {shape}")
    if not print_all and len(rows) > top_k:
        lines.append(f"  ... and {len(rows) - top_k:,} more trainable tensors")

    return "\n".join(lines)


def get_hidden_size(model):
    """
    Robustly retrieves the hidden dimension (d_model) from a Hugging Face model.
    Works for Llama, Gemma, Gemma 3 (multimodal), and most other architectures.
    """
    config = model.config

    # 1. Try standard top-level attributes
    for attr in ["hidden_size", "dim", "d_model", "model_dim"]:
        if hasattr(config, attr):
            return getattr(config, attr)

    # 2. Try nested configs (common in multimodal models like Gemma 3)
    # Most Gemma 3 configs store text-specific params in 'text_config'
    if hasattr(config, "text_config"):
        t_config = config.text_config
        for attr in ["hidden_size", "dim", "d_model"]:
            if hasattr(t_config, attr):
                return getattr(t_config, attr)

    # 3. Ultimate Fallback: Inspect the actual embedding layer
    # This works regardless of naming conventions in the config
    try:
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings().weight.shape[-1]

        # Manually search modules if helper doesn't exist
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                return module.weight.shape[-1]
    except Exception as e:
        print(f"Warning: Could not extract dimension from embeddings: {e}")

    raise AttributeError(
        f"Could not determine hidden size for model type: {type(model).__name__}. "
        "Please check the model config structure."
    )
