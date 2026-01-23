import os

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_PROMPT_TMPL = "Input:\n{text}\n\nPredict a score from 1 to 4:"


class FrozenLlama:
    """Wrapper around a frozen decoder-only LLaMA backbone."""

    def __init__(self, model_name: str, device: str):
        hf_home = os.environ.get("HF_HOME", None)
        cache_kwargs = {"cache_dir": hf_home} if hf_home else {}
        self.tok = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, **cache_kwargs
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16, **cache_kwargs
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)
        self.device = device

    def encode(self, texts, max_length: int = 256, already_prompted: bool = False):
        """
        Tokenize and encode a list of prompts.
        If already_prompted=False, wraps raw text with the default template.
        """
        if already_prompted:
            prompts = list(texts)
        else:
            prompts = [DEFAULT_PROMPT_TMPL.format(text=t) for t in texts]
        batch = self.tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch, output_hidden_states=True)
        return out.hidden_states[-1]  # [B, S, d_model]
