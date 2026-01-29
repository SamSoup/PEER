# PEER: Three-Stage Prototype Regressor

End-to-end pipeline mirroring the provided spec, adapted to the existing `/data` loaders and dataset→prompt mapping from `data/datasets.py`. Uses dataset-specific prompts automatically for STS/translation datasets.

## Prereqs
- GPU with enough memory for Meta-Llama-3.1-8B-Instruct.
- HF access to `meta-llama/Meta-Llama-3.1-8B-Instruct` (set `HF_HOME`, `HF_TOKEN` as needed).
- Writable cache area (examples below use `/scratch/06782/ysu707/`).

# Metrics and explanations are written under save_dir:
#   metrics.json         -> mse, rmse, pearson, spearman, kendall on test split
#   explanations.json    -> 5 sampled test examples with prototype contributions

## Hyperparameter defaults (CLI mappable)
- Stage A: `d_h=256`, `m=8`, `mq=8`, `n_heads=8`, `num_layers=3`, `lambda_emb=0.1`, `huber_delta=0.5`, `lr=1e-3`, `weight_decay=0.01`, `epochs=5`, `batch_size=16`, `max_length=256`.
- Stage B: `K=128`, `T=512`, `tau_start=1.0`, `tau_final=0.1`, `lambda_ov=1.0`, `lambda_rep=0.1`, `huber_delta=0.5`, `lr=3e-4`, `weight_decay=0.01`, `epochs=10`, `batch_size=16`, `max_length=256`, `margin=0.2`.
- Stage C: `epochs=2`, `batch_size=16`, `lr=3e-5`, `weight_decay=0.01`, `max_length=256`, `huber_delta=1.0`.

## Notes
- Prompts are built per-dataset using the same templates in `data/datasets.py` (pair-aware; falls back to a generic prompt when no mapping is found). Raw pair inputs (`sentence1`, `sentence2`) are preserved by default (`combine_fields=False`).
- Validation reporting now mirrors `train.py` metrics: `mse`, `rmse`, `pearson`, `spearman`, `kendall` (Spearman/Kendall may be `nan` if the dependency is unavailable).
- Cache artifacts are written under `cache_dir`: `cache_mem.pt`, `cache_keys.pt`, `cache_y.pt`, and `cache_text.json` (human-readable text pairs).
- Debugging helpers: outputs are unbounded; validation prints `std/min/max/mean(yhat)`, mean(y), and sigmoid saturation % to detect constant predictors. Explanations/metrics are saved at inference.
