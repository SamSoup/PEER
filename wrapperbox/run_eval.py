# run_eval.py

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# labels via your registry-backed raw datamodule
from data.factory import build_pairwise_datamodule


# ----------------------------
# metrics
# ----------------------------
def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(_mse(y_true, y_pred)))


def _corr_safe(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    scipy stats functions return nan if constant arrays; we keep nan.
    """
    try:
        v = fn(y_true, y_pred)
        # pearsonr returns (r, p); spearmanr returns (rho, p); kendalltau returns (tau, p)
        if isinstance(v, tuple):
            return float(v[0])
        return float(v)
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": _mse(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "pearson": _corr_safe(lambda a, b: pearsonr(a, b), y_true, y_pred),
        "spearman": _corr_safe(lambda a, b: spearmanr(a, b), y_true, y_pred),
        "kendall": _corr_safe(lambda a, b: kendalltau(a, b), y_true, y_pred),
    }


# ----------------------------
# loading
# ----------------------------
def load_embeddings(
    dir_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(dir_path / "train.npy")
    val = np.load(dir_path / "validation.npy")
    test = np.load(dir_path / "test.npy")
    if train.ndim != 2 or val.ndim != 2 or test.ndim != 2:
        raise ValueError("Expected embeddings to be 2D arrays (N, D).")
    if train.shape[1] != val.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("Embedding dimension mismatch across splits.")
    return train, val, test


def load_labels_via_datamodule(
    dataset_name: str,
    model_name_for_dm: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild raw datamodule and read labels in dataset order.

    model_name_for_dm is only used to initialize tokenizer in base dm; irrelevant for raw labels,
    but build_pairwise_datamodule requires it. You can pass the embedding model name.
    """
    dm = build_pairwise_datamodule(
        dataset_name=dataset_name,
        model_name=model_name_for_dm,
        max_seq_length=8,  # irrelevant for raw labels path
        batch_size=32,
        tokenize_inputs=False,
        combine_fields=False,
    )
    if dm is None:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    dm.setup(None)

    def _labels(split: str) -> np.ndarray:
        if split not in dm.dataset:
            raise ValueError(
                f"Missing split '{split}' in dataset for {dataset_name}"
            )
        ys = [float(ex["score"]) for ex in dm.dataset[split]]
        return np.asarray(ys, dtype=np.float32)

    return _labels("train"), _labels("validation"), _labels("test")


# ----------------------------
# model selection (train->val)
# ----------------------------
def select_best_on_val(
    model_name: str,
    candidates: List[Tuple[Dict[str, Any], Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    candidates: list of (config_dict, estimator_instance)
    Selection criterion: minimize val MSE.
    """
    best = {
        "model": model_name,
        "best_config": None,
        "val_metrics": None,
        "val_mse": float("inf"),
    }

    total = len(candidates)
    for i, (cfg, est) in enumerate(candidates, 1):
        if verbose:
            print(f"[{model_name}] {i}/{total} cfg={cfg}")

        est.fit(X_train, y_train)
        pred = est.predict(X_val)
        m = compute_metrics(y_val, pred)
        if m["mse"] < best["val_mse"]:
            best["val_mse"] = m["mse"]
            best["best_config"] = cfg
            best["val_metrics"] = m

    return best


def refit_and_test(
    model_name: str,
    best_config: Dict[str, Any],
    make_estimator_fn,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    est = make_estimator_fn(best_config)
    est.fit(X_trainval, y_trainval)
    pred = est.predict(X_test)
    test_metrics = compute_metrics(y_test, pred)
    return {
        "model": model_name,
        "best_config": best_config,
        "test_metrics": test_metrics,
    }


# ----------------------------
# candidate grids
# ----------------------------
def knn_candidates() -> List[Tuple[Dict[str, Any], Any]]:
    ks = [3, 5, 7, 9, 11, 50, 100, 200, 500, 1000]
    weights = ["uniform", "distance"]
    metrics = ["euclidean", "cosine"]

    out: List[Tuple[Dict[str, Any], Any]] = []
    for k in ks:
        for w in weights:
            for m in metrics:
                cfg = {"n_neighbors": k, "weights": w, "metric": m}
                est = KNeighborsRegressor(**cfg)
                out.append((cfg, est))
    return out


def dt_candidates(random_state: int = 0) -> List[Tuple[Dict[str, Any], Any]]:
    depths = [None, 5, 10, 20]
    out: List[Tuple[Dict[str, Any], Any]] = []
    for d in depths:
        cfg = {"max_depth": d, "random_state": random_state}
        est = DecisionTreeRegressor(**cfg)
        out.append((cfg, est))
    return out


def rf_candidates(random_state: int = 0) -> List[Tuple[Dict[str, Any], Any]]:
    n_estimators = [10, 100, 200]
    depths = [None, 5, 10, 20]
    out: List[Tuple[Dict[str, Any], Any]] = []
    for n in n_estimators:
        for d in depths:
            cfg = {
                "n_estimators": n,
                "max_depth": d,
                "random_state": random_state,
                "n_jobs": -1,
            }
            est = RandomForestRegressor(**cfg)
            out.append((cfg, est))
    return out


# ----------------------------
# CLI / main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate embeddings with kNN / DecisionTree / RandomForest"
    )
    p.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help="Directory containing train.npy, validation.npy, test.npy (from generate_embeddings)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for results.json",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset registry name (same as used for embeddings)",
    )
    p.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help="Model name used for embeddings (only used here to rebuild dm for labels)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for tree/forest",
    )
    p.add_argument(
        "--verbose_cv",
        action="store_true",
        help="Print progress for each hyperparam setting",
    )
    return p.parse_args()


def main():
    args = parse_args()
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test = load_embeddings(emb_dir)
    y_train, y_val, y_test = load_labels_via_datamodule(
        dataset_name=args.dataset,
        model_name_for_dm=args.llm_name,
    )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"train size mismatch: X={X_train.shape[0]} y={y_train.shape[0]}"
        )
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError(
            f"validation size mismatch: X={X_val.shape[0]} y={y_val.shape[0]}"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"test size mismatch: X={X_test.shape[0]} y={y_test.shape[0]}"
        )

    # Selection: train -> val
    knn_best = select_best_on_val(
        "knn",
        knn_candidates(),
        X_train,
        y_train,
        X_val,
        y_val,
        verbose=args.verbose_cv,
    )

    dt_best = select_best_on_val(
        "decision_tree",
        dt_candidates(random_state=args.seed),
        X_train,
        y_train,
        X_val,
        y_val,
        verbose=args.verbose_cv,
    )

    rf_best = select_best_on_val(
        "random_forest",
        rf_candidates(random_state=args.seed),
        X_train,
        y_train,
        X_val,
        y_val,
        verbose=args.verbose_cv,
    )

    # Final train: train+val -> test
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    def make_knn(cfg):
        return KNeighborsRegressor(**cfg)

    def make_dt(cfg):
        return DecisionTreeRegressor(**cfg)

    def make_rf(cfg):
        return RandomForestRegressor(**cfg)

    knn_test = refit_and_test(
        "knn",
        knn_best["best_config"],
        make_knn,
        X_trainval,
        y_trainval,
        X_test,
        y_test,
    )
    dt_test = refit_and_test(
        "decision_tree",
        dt_best["best_config"],
        make_dt,
        X_trainval,
        y_trainval,
        X_test,
        y_test,
    )
    rf_test = refit_and_test(
        "random_forest",
        rf_best["best_config"],
        make_rf,
        X_trainval,
        y_trainval,
        X_test,
        y_test,
    )

    payload: Dict[str, Any] = {
        "dataset": args.dataset,
        "llm_name": args.llm_name,
        "emb_dir": str(emb_dir),
        "results": {
            "knn": {
                "best_config": knn_best["best_config"],
                "val_metrics": knn_best["val_metrics"],
                "test_metrics": knn_test["test_metrics"],
            },
            "decision_tree": {
                "best_config": dt_best["best_config"],
                "val_metrics": dt_best["val_metrics"],
                "test_metrics": dt_test["test_metrics"],
            },
            "random_forest": {
                "best_config": rf_best["best_config"],
                "val_metrics": rf_best["val_metrics"],
                "test_metrics": rf_test["test_metrics"],
            },
        },
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print("Saved:", out_path)

    # Print a quick summary
    for name in ["knn", "decision_tree", "random_forest"]:
        tm = payload["results"][name]["test_metrics"]
        print(
            f"{name:>14} test mse={tm['mse']:.6f} rmse={tm['rmse']:.6f} "
            f"pearson={tm['pearson']:.4f} spearman={tm['spearman']:.4f} kendall={tm['kendall']:.4f}"
        )


if __name__ == "__main__":
    main()
