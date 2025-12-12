from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_val_test(
    df: pd.DataFrame,
    train: float,
    val: float,
    test: float,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Stratified (optional) train/val/test split.

    The function is deterministic given `seed`.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    df_train, df_temp = train_test_split(df, test_size=(1.0 - train), random_state=seed, stratify=stratify)

    # Now split temp into val/test
    remaining = val + test
    val_ratio = val / remaining if remaining > 0 else 0.0

    stratify_temp = df_temp[stratify_col] if stratify_col and stratify_col in df_temp.columns else None
    df_val, df_test = train_test_split(df_temp, test_size=(1.0 - val_ratio), random_state=seed, stratify=stratify_temp)

    return {"train": df_train.reset_index(drop=True), "val": df_val.reset_index(drop=True), "test": df_test.reset_index(drop=True)}
