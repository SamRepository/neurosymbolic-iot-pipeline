from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _safe_stratify(df: pd.DataFrame, stratify_col: Optional[str]) -> Optional[pd.Series]:
    if not stratify_col or stratify_col not in df.columns:
        return None
    vc = df[stratify_col].value_counts(dropna=False)
    # Need at least 2 classes and each class at least 2 samples for stratify to be valid.
    if vc.shape[0] < 2:
        return None
    if vc.min() < 2:
        return None
    return df[stratify_col]


def split_train_val_test(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        raise ValueError("split_train_val_test received an empty dataframe.")

    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        train, val, test = train / total, val / total, test / total

    n = len(df)
    # If too small, skip splitting to avoid hard crash
    if n < 3:
        return {"train": df.copy(), "val": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}

    stratify = _safe_stratify(df, stratify_col)

    # First split: train vs temp
    test_size_1 = max(1, int(round((1.0 - train) * n)))
    test_size_1 = min(test_size_1, n - 1)  # ensure train not empty
    df_train, df_temp = train_test_split(
        df,
        test_size=test_size_1,
        random_state=seed,
        stratify=stratify,
    )

    if len(df_temp) < 2 or (val == 0 and test == 0):
        return {"train": df_train.reset_index(drop=True), "val": df_temp.iloc[0:0].copy(), "test": df_temp.reset_index(drop=True)}

    # Second split: val vs test (within temp)
    # proportion of val within (val+test)
    val_ratio = val / (val + test) if (val + test) > 0 else 0.5

    n_temp = len(df_temp)
    val_size = max(1, int(round(val_ratio * n_temp)))
    val_size = min(val_size, n_temp - 1)  # ensure test not empty

    stratify_temp = _safe_stratify(df_temp, stratify_col)

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(n_temp - val_size),
        random_state=seed,
        stratify=stratify_temp,
    )

    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
