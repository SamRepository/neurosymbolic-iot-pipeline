from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SphereWindow:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    label: str
    seq: np.ndarray  # shape [T, F]


def _infer_numeric_feature_cols(df: pd.DataFrame, exclude: set[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _resample_to_len(x: np.ndarray, seq_len: int) -> np.ndarray:
    """
    x: [N, F]
    returns [seq_len, F] by uniform index sampling + padding if needed.
    """
    n = x.shape[0]
    if n == 0:
        return np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    if n >= seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(int)
        return x[idx].astype(np.float32, copy=False)
    # pad by repeating last row
    pad = np.repeat(x[-1:, :], repeats=(seq_len - n), axis=0)
    out = np.concatenate([x, pad], axis=0)
    return out.astype(np.float32, copy=False)


def build_sphere_labeled_stream(cfg: Dict) -> pd.DataFrame:
    """
    Reuses your preprocess_sphere helpers:
      - _load_label_matrix
      - _load_stream
      - _attach_labels_nearest
    Returns: DataFrame with timestamp + numeric features + label
    """
    from neurosymbolic_iot.data_processing.preprocess_sphere import (
        _attach_labels_nearest,
        _load_label_matrix,
        _load_stream,
    )

    ds = cfg["datasets"]["sphere"]
    tolerance_s = float(ds.get("label_join_tolerance_seconds", 5.0))

    labels, _meta_labels = _load_label_matrix(cfg)
    stream, _meta_stream = _load_stream(cfg)

    labeled = _attach_labels_nearest(stream, labels, tolerance_s=tolerance_s)
    if labeled.empty:
        raise RuntimeError("SPHERE: label join produced empty labeled stream. Increase tolerance or pick another stream.")
    return labeled


def build_sphere_windows(
    labeled: pd.DataFrame,
    *,
    window_seconds: int,
    stride_seconds: int,
    min_rows: int,
    seq_len: int,
) -> List[SphereWindow]:
    labeled = labeled.sort_values("timestamp").reset_index(drop=True)

    feat_cols = _infer_numeric_feature_cols(labeled, exclude={"timestamp", "label"})
    if not feat_cols:
        labeled["dummy_feature"] = 1.0
        feat_cols = ["dummy_feature"]

    start = labeled["timestamp"].min()
    end = labeled["timestamp"].max()

    win = pd.Timedelta(seconds=int(window_seconds))
    stride = pd.Timedelta(seconds=int(stride_seconds))

    windows: List[SphereWindow] = []
    t = start
    while t + win <= end:
        w = labeled[(labeled["timestamp"] >= t) & (labeled["timestamp"] < t + win)]
        if len(w) >= int(min_rows):
            mode = w["label"].mode(dropna=True)
            if not mode.empty:
                label = str(mode.iloc[0])
                x = w[feat_cols].to_numpy(dtype=np.float32, copy=False)
                x = _resample_to_len(x, seq_len=seq_len)
                windows.append(SphereWindow(start_time=t, end_time=t + win, label=label, seq=x))
        t = t + stride

    if not windows:
        raise RuntimeError("SPHERE: no windows created. Reduce min_rows or adjust window/stride.")
    return windows
