from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

log = logging.getLogger(__name__)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def save_confusion_csv(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_csv: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ensure_dir(out_csv.parent)
    import pandas as pd

    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(out_csv, index=True)


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class SplitFrames:
    train: Any
    val: Any
    test: Any


def safe_split_df(df, train: float, val: float, test: float, seed: int, stratify_col: Optional[str] = "label"):
    """
    Uses your existing split utility and falls back when stratification is impossible
    due to rare classes.
    """
    from neurosymbolic_iot.data_processing.splits import split_train_val_test

    try:
        splits = split_train_val_test(
            df,
            train=train,
            val=val,
            test=test,
            seed=seed,
            stratify_col=stratify_col,
        )
        return SplitFrames(train=splits["train"], val=splits["val"], test=splits["test"])
    except Exception as e:
        log.warning("Stratified split failed (%s). Falling back to non-stratified split.", e)
        splits = split_train_val_test(
            df,
            train=train,
            val=val,
            test=test,
            seed=seed,
            stratify_col=None,
        )
        return SplitFrames(train=splits["train"], val=splits["val"], test=splits["test"])
