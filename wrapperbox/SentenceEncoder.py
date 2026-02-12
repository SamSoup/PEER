# /wrapperbox/SentenceEncoder.py

from __future__ import annotations

import os
from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


def _get_hidden_dim(config) -> int:
    """
    Robustly infer text hidden dim across:
      - plain LMs (hidden_size / n_embd / d_model / etc.)
      - multimodal wrappers (config.text_config / config.language_config)
      - nested dict-like configs (recursive search)
    """

    CAND_KEYS = (
        "hidden_size",
        "n_embd",
        "d_model",
        "model_dim",
        "dim",
        "embedding_size",
    )

    def _try_one(cfg) -> int | None:
        for k in CAND_KEYS:
            v = (
                getattr(cfg, k, None)
                if not isinstance(cfg, dict)
                else cfg.get(k)
            )
            if isinstance(v, int) and v > 0:
                return v
        return None

    def _to_dict(cfg):
        if isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "to_dict"):
            try:
                return cfg.to_dict()
            except Exception:
                return None
        return None

    # 1) direct
    v = _try_one(config)
    if v is not None:
        return v

    # 2) common nested configs
    for child_key in ("text_config", "language_config"):
        child = getattr(config, child_key, None)
        if child is None and isinstance(config, dict):
            child = config.get(child_key)
        if child is not None:
            v = _try_one(child)
            if v is not None:
                return v

    # 3) recursive search in dict representation (handles weird wrappers)
    d = _to_dict(config)
    if isinstance(d, dict):

        def _recurse(obj) -> int | None:
            if isinstance(obj, dict):
                v0 = _try_one(obj)
                if v0 is not None:
                    return v0
                for vv in obj.values():
                    out = _recurse(vv)
                    if out is not None:
                        return out
            elif isinstance(obj, (list, tuple)):
                for vv in obj:
                    out = _recurse(vv)
                    if out is not None:
                        return out
            return None

        v = _recurse(d)
        if v is not None:
            return v

    raise AttributeError(
        f"Could not infer hidden dim from config. Top-level keys: "
        f"{list(d.keys())[:60] if isinstance(d, dict) else str(type(config))}"
    )


class SentenceEncoder(nn.Module):
    """
    Encode list[str] -> Tensor[B, D] using mean pooling of last_hidden_state.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "/scratch/06782/ysu707/.cache",
        normalize_embeddings: bool = False,
        batch_size: int = 32,
        max_tokens: int = 4096,
        dtype: str = "bf16",  # bf16|fp16|fp32
    ):
        super().__init__()

        os.environ["HF_HOME"] = cache_dir

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.max_tokens = int(max_tokens)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token or "[PAD]"
            )

        # right padding is simpler for mean pooling
        self.tokenizer.padding_side = "right"

        base = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        # pick a text backbone if model is a wrapper
        text_backbone = (
            getattr(base, "text_model", None)
            or getattr(base, "language_model", None)
            or getattr(base, "model", None)
            or base
        )
        self.model = text_backbone
        self.model.eval()

        # hidden dim: try config, then fallback to a tiny forward pass
        try:
            self.hidden_dim = _get_hidden_dim(
                getattr(text_backbone, "config", getattr(base, "config", None))
            )
        except Exception:
            self.hidden_dim = self._infer_hidden_dim_by_forward()

    def _infer_hidden_dim_by_forward(self) -> int:
        """Ultimate fallback: run one tiny forward pass and read last_hidden_state dim."""
        with torch.no_grad():
            enc = self.tokenizer(
                ["hi"],
                padding=True,
                truncation=True,
                max_length=min(8, self.max_tokens),
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            hs = out.last_hidden_state
            return int(hs.shape[-1])

    def _truncate_texts(self, texts: List[str]) -> List[str]:
        out: List[str] = []
        for t in texts:
            t = (t or "").replace("\n", " ")
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) > self.max_tokens:
                ids = ids[: self.max_tokens]
                t = self.tokenizer.decode(ids, skip_special_tokens=True)
            out.append(t)
        return out

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, self.hidden_dim, device=self.device)

        texts = [str(t) for t in texts]
        texts = self._truncate_texts(texts)

        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,  # ✅ fixes HF "no max length" warning
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            hs = out.last_hidden_state  # [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)  # [B, T, 1]

            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if self.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            all_vecs.append(pooled)

        return torch.cat(all_vecs, dim=0)

    def encode(self, texts: List[str]) -> torch.Tensor:
        return self.forward(texts)
