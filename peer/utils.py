import torch
import torch.nn.functional as F
import torchmetrics.functional as tmf


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
