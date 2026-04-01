from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from neurosymbolic_iot.neural_perception.trainer import evaluate_classifier, train_loop
from neurosymbolic_iot.neural_perception.utils import compute_metrics, ensure_dir, save_json

log = logging.getLogger(__name__)


def run_kfold_cv(
    *,
    build_dataset_fn: Callable[
        [pd.DataFrame, pd.DataFrame, Dict[str, int]],
        Tuple[DataLoader, DataLoader],
    ],
    build_model_fn: Callable[[int], torch.nn.Module],
    df_all: pd.DataFrame,
    label_col: str,
    id2label: List[str],
    cfg: Dict[str, Any],
    out_dir: Path,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Stratified k-fold cross-validation for neural perception models.

    Paper Section 4.3 claims 5-fold CV with ±1.2% variance.

    Parameters
    ----------
    build_dataset_fn : callable(train_df, val_df, label2id) -> (train_loader, val_loader)
        Builds DataLoaders from split DataFrames.
    build_model_fn : callable(num_classes) -> nn.Module
        Returns a fresh (untrained) model instance.
    df_all : DataFrame
        Full dataset with at least `label_col` column.
    label_col : str
        Column name for stratification labels.
    id2label : list of str
        Ordered class labels.
    cfg : dict
        Full pipeline config (used for training hyperparams).
    out_dir : Path
        Output directory for CV results.
    k : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with per-fold metrics and aggregate mean ± std.
    """
    ensure_dir(out_dir)

    label2id = {lab: i for i, lab in enumerate(id2label)}
    device = str(cfg.get("neural_perception", {}).get("device", "auto"))
    from neurosymbolic_iot.neural_perception.utils import pick_device
    device = pick_device(device)

    np_cfg = cfg.get("neural_perception", {})
    epochs = int(np_cfg.get("casas", {}).get("epochs", np_cfg.get("sphere", {}).get("epochs", 30)))
    lr = float(np_cfg.get("casas", {}).get("lr", np_cfg.get("sphere", {}).get("lr", 1e-3)))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    labels = df_all[label_col].astype(str).values

    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_all, labels)):
        fold_dir = out_dir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        df_train = df_all.iloc[train_idx].reset_index(drop=True)
        df_val = df_all.iloc[val_idx].reset_index(drop=True)

        log.info("Fold %d/%d: train=%d, val=%d", fold_idx + 1, k, len(df_train), len(df_val))

        train_loader, val_loader = build_dataset_fn(df_train, df_val, label2id)

        model = build_model_fn(len(id2label))

        result = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,  # In CV, val fold serves as test
            device=device,
            id2label=id2label,
            out_dir=fold_dir,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )

        fold_result = {
            "fold": fold_idx,
            "best_epoch": result.best_epoch,
            "val_accuracy": result.metrics.get("test", {}).get("accuracy", 0.0),
            "val_f1_macro": result.metrics.get("test", {}).get("f1_macro", 0.0),
            "val_f1_weighted": result.metrics.get("test", {}).get("f1_weighted", 0.0),
        }
        fold_metrics.append(fold_result)
        log.info(
            "Fold %d: acc=%.4f, f1_macro=%.4f, f1_weighted=%.4f",
            fold_idx,
            fold_result["val_accuracy"],
            fold_result["val_f1_macro"],
            fold_result["val_f1_weighted"],
        )

    # Aggregate
    accs = [f["val_accuracy"] for f in fold_metrics]
    f1_macros = [f["val_f1_macro"] for f in fold_metrics]
    f1_weights = [f["val_f1_weighted"] for f in fold_metrics]

    summary = {
        "k": k,
        "seed": seed,
        "folds": fold_metrics,
        "aggregate": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1_macros)),
            "f1_macro_std": float(np.std(f1_macros)),
            "f1_weighted_mean": float(np.mean(f1_weights)),
            "f1_weighted_std": float(np.std(f1_weights)),
        },
    }

    save_json(out_dir / "cv_results.json", summary)
    log.info(
        "%d-fold CV: accuracy=%.4f ± %.4f, F1-macro=%.4f ± %.4f",
        k,
        summary["aggregate"]["accuracy_mean"],
        summary["aggregate"]["accuracy_std"],
        summary["aggregate"]["f1_macro_mean"],
        summary["aggregate"]["f1_macro_std"],
    )
    return summary
