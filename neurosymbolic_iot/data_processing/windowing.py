from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


def majority_vote(labels: pd.Series) -> Optional[str]:
    labels = labels.dropna()
    if labels.empty:
        return None
    return labels.value_counts().idxmax()


def build_event_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create bag-of-events features: counts of (sensor,value) pairs."""
    key = df["sensor"].astype(str) + "=" + df["value"].astype(str)
    counts = key.value_counts()
    # Wide, sparse-ish dict for a single window
    return pd.DataFrame([counts.to_dict()])


def rolling_event_windows(
    events: pd.DataFrame,
    window_minutes: int,
    stride_minutes: int,
    min_events: int = 1,
    label_mode: str = "majority",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert event stream into rolling time windows.

    Returns:
      X: feature rows per window (wide DataFrame)
      Y: window metadata + label per window
    """
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    events = events.sort_values("timestamp").reset_index(drop=True)
    t0 = events["timestamp"].min()
    t1 = events["timestamp"].max()

    w = timedelta(minutes=window_minutes)
    s = timedelta(minutes=stride_minutes)

    X_rows: List[pd.DataFrame] = []
    meta_rows: List[Dict[str, object]] = []

    start = t0
    idx = 0
    while start <= t1:
        end = start + w
        wdf = events[(events["timestamp"] >= start) & (events["timestamp"] < end)]
        if len(wdf) >= min_events:
            feats = build_event_count_features(wdf)
            feats.insert(0, "window_id", idx)
            X_rows.append(feats)

            label = None
            if label_mode == "majority" and "activity" in wdf.columns:
                label = majority_vote(wdf["activity"])
            elif label_mode == "last_event" and "activity" in wdf.columns:
                label = wdf["activity"].dropna().iloc[-1] if not wdf["activity"].dropna().empty else None

            meta_rows.append(
                {
                    "window_id": idx,
                    "start_time": start,
                    "end_time": end,
                    "n_events": int(len(wdf)),
                    "label": label,
                }
            )
            idx += 1
        start = start + s

    X = pd.concat(X_rows, ignore_index=True) if X_rows else pd.DataFrame()
    Y = pd.DataFrame(meta_rows)
    return X, Y
