"""
Generate and cache embeddings for a given dataset using the existing data module.

Usage:
  python generate_embeddings.py --dataset stsb --encoder meta-llama/Llama-3.1-8B-Instruct --output-dir embeddings/stsb_llama3.1

Notes:
- Relies on data.get_datamodule to build the dataset.
- Uses EmbeddingDataModule to cache embeddings under output-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

# Allow running from repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import get_datamodule, EmbeddingDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached embeddings for a dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name (per data modules)")
    parser.add_argument("--encoder", required=True, help="HF encoder name")
    parser.add_argument(
        "--encoder-type",
        default="sentence",
        choices=["sentence", "bert", "average", "mean"],
        help="Encoder factory type (default: sentence)",
    )
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--combine-fields", action="store_true")
    parser.add_argument("--combine-separator-token", default="[SEP]")
    parser.add_argument("--output-dir", required=True, help="Directory to store cached embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.encoders.factory import get_encoder

    encoder = get_encoder(
        encoder_type=args.encoder_type,
        model_name=args.encoder,
    ).to(device)
    encoder.eval()

    raw_dm = get_datamodule(
        dataset_name=args.dataset,
        model_name=args.encoder,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        tokenize_inputs=False,
        combine_fields=args.combine_fields,
        combine_separator_token=args.combine_separator_token,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_dm = EmbeddingDataModule(
        raw_dm=raw_dm,
        encoder=encoder,
        embedding_cache_dir=str(out_dir),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        device=device,
    )
    embed_dm.setup("fit")
    embed_dm.setup("test")
    print(f"Embeddings cached to {out_dir}")


if __name__ == "__main__":
    main()
