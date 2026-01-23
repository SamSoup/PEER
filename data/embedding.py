# data/embedding.py

import os
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm


class _EmbeddingTensorDataset(Dataset):
    """
    Tiny dataset that returns precomputed embeddings + labels directly.
    __getitem__ returns:
        {
            "embeddings": embeddings[i],  # (H,)
            "labels":     labels[i],      # scalar
        }
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        assert embeddings.shape[0] == labels.shape[0]
        self.embeddings = embeddings  # (N,H) float32 CPU or GPU agnostic
        self.labels = labels  # (N,)   float32
        # Keep track of original row indices so downstream models can know
        # which training example each batch element came from (needed for
        # anchor-routing in PRLE v2).
        self.indices = torch.arange(labels.shape[0], dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embeddings": self.embeddings[idx],  # (H,)
            "labels": self.labels[idx],  # (1)
            "indices": self.indices[idx],  # (1)
        }


class EmbeddingDataModule(pl.LightningDataModule):
    """
    Wraps an *existing* text/token DataModule (e.g. STSBDataModule),
    plus a frozen encoder.

    Responsibilities:
      - On setup(), ensure raw_dm is set up (so .dataset[...] exists).
      - For each split, compute (or load) embeddings with the frozen encoder
        and pair them with the ground-truth labels.
      - Serve train/val/test loaders that emit batches of embeddings+labels.

    Caching:
      - Embeddings for each split are saved to:
            {embedding_cache_dir}/{split}_embeds.npy  (shape [N,H], float32)
      - We DO NOT save labels or raw text locally, per instructions.
        Labels are reloaded from the HF dataset each run.
    """

    def __init__(
        self,
        raw_dm: pl.LightningDataModule,
        encoder: torch.nn.Module,
        embedding_cache_dir: str,
        train_batch_size: int,
        eval_batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.raw_dm = raw_dm
        self.encoder = encoder
        self.embedding_cache_dir = embedding_cache_dir

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size or train_batch_size

        # LightningDataModule contract: we'll populate these in setup()
        self.train_embeddings: Optional[torch.Tensor] = None
        self.val_embeddings: Optional[torch.Tensor] = None
        self.test_embeddings: Optional[torch.Tensor] = None

        self.train_labels: Optional[torch.Tensor] = None
        self.val_labels: Optional[torch.Tensor] = None
        self.test_labels: Optional[torch.Tensor] = None

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

        # device for temporary encoder forward (cuda vs cpu)
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = device

    # -------------------------
    # internal helpers
    # -------------------------

    def _build_ordered_loader(
        self, split_name: str, batch_size: int
    ) -> DataLoader:
        """
        Build a deterministic, NON-shuffled loader over raw_dm.dataset[split_name]
        so we can walk the split once in a fixed order.
        The raw_dm may (a) return raw 'text' strings/tuples or
        (b) return tokenized tensors. We reuse raw_dm's collator logic.
        """
        ds = self.raw_dm.dataset[split_name]

        # If raw_dm handles raw text, it usually sets tokenize_inputs=False
        # and defines _collate_raw for tuples/strings.
        collate = (
            None
            if getattr(self.raw_dm, "tokenize_inputs", True)
            else getattr(self.raw_dm, "_collate_raw", None)
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # <-- crucial for consistent caching
            collate_fn=collate,
        )

    @torch.no_grad()
    def _gather_labels_only(self, split_name: str) -> torch.Tensor:
        """
        Get the labels for a split in the same order as the HF dataset rows.
        This is cheap -- just iterate once without running the encoder.
        """
        loader = self._build_ordered_loader(split_name, self.eval_batch_size)
        all_labels = []
        for batch in loader:
            all_labels.append(batch["labels"].detach().cpu())
        labels = torch.cat(all_labels, dim=0).float()  # (N,)
        return labels

    @torch.no_grad()
    def _compute_or_load_embeddings(
        self,
        split_name: str,
    ) -> torch.Tensor:
        """
        Core routine:
          - if {split}_embeds.npy exists in cache_dir, load it
          - else, iterate over raw_dm's split with frozen encoder and save it.
        Returns a torch.FloatTensor (N,H) on CPU.
        """
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        npy_path = os.path.join(
            self.embedding_cache_dir, f"{split_name}_embeds.npy"
        )

        # If cached, just load
        if os.path.exists(npy_path):
            arr = np.load(npy_path)  # shape (N,H), float32
            return torch.from_numpy(arr).float()

        # Otherwise compute embeddings now
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        ordered_loader = self._build_ordered_loader(
            split_name,
            # use train_batch_size for train, eval_batch_size otherwise
            (
                self.train_batch_size
                if split_name == "train"
                else self.eval_batch_size
            ),
        )

        all_embeds: List[torch.Tensor] = []

        # Check mode (raw text or tokenized) by peeking at first batch
        first_batch = next(iter(ordered_loader))
        raw_mode = "text" in first_batch  # True => pass list/tuple of strings

        # restart loader after peek
        ordered_loader = self._build_ordered_loader(
            split_name,
            (
                self.train_batch_size
                if split_name == "train"
                else self.eval_batch_size
            ),
        )

        for batch in tqdm(ordered_loader, desc=f"[encode:{split_name}]"):
            if raw_mode:
                enc_in = {"text": batch["text"]}
                with torch.no_grad():
                    emb = self.encoder(enc_in["text"])
            else:
                enc_in = {"input_ids": batch["input_ids"].to(self.device)}
                if "attention_mask" in batch:
                    enc_in["attention_mask"] = batch["attention_mask"].to(
                        self.device
                    )
                if "token_type_ids" in batch:
                    enc_in["token_type_ids"] = batch["token_type_ids"].to(
                        self.device
                    )
                with torch.no_grad():
                    emb = self.encoder(**enc_in)

            all_embeds.append(emb.detach().cpu())

        embeds = torch.cat(all_embeds, dim=0).float()  # (N,H)

        # cache to disk
        np.save(npy_path, embeds.numpy().astype(np.float32))

        return embeds

    # -------------------------
    # LightningDataModule hooks
    # -------------------------

    def setup(self, stage: Optional[str] = None):
        """
        1. Ensure raw_dm has loaded the HF dataset and formatted labels.
        2. For each split, build (or load) embeddings and gather labels.
        3. Create split-specific in-memory datasets for downstream loaders.
        """
        # Make sure raw_dm has its dataset loaded.
        # raw_dm.setup("fit") usually prepares train/validation,
        # raw_dm.setup("test") prepares test.
        self.raw_dm.setup("fit")
        self.raw_dm.setup("test")

        # ---- TRAIN SPLIT ----
        train_emb = self._compute_or_load_embeddings("train")  # (N_train,H)
        train_lab = self._gather_labels_only("train")  # (N_train,)
        self.train_embeddings = train_emb
        self.train_labels = train_lab

        # ---- VALIDATION SPLIT ----
        # different data modules call it "val" or "validation";
        # in STSBDataModule it is "validation".
        val_split_name = "validation"
        if val_split_name not in self.raw_dm.dataset:
            # fallback: "val"
            val_split_name = "val"

        val_emb = self._compute_or_load_embeddings(val_split_name)  # (N_val,H)
        val_lab = self._gather_labels_only(val_split_name)  # (N_val,)
        self.val_embeddings = val_emb
        self.val_labels = val_lab

        # ---- TEST SPLIT ----
        test_split_name = "test"
        test_emb = self._compute_or_load_embeddings(
            test_split_name
        )  # (N_test,H)
        test_lab = self._gather_labels_only(test_split_name)  # (N_test,)
        self.test_embeddings = test_emb
        self.test_labels = test_lab

        # build dataset objects
        self._train_ds = _EmbeddingTensorDataset(
            embeddings=self.train_embeddings,
            labels=self.train_labels,
        )
        self._val_ds = _EmbeddingTensorDataset(
            embeddings=self.val_embeddings,
            labels=self.val_labels,
        )
        self._test_ds = _EmbeddingTensorDataset(
            embeddings=self.test_embeddings,
            labels=self.test_labels,
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )
