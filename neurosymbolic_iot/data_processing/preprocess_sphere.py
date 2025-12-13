# (PASTE THE FULL preprocess_sphere.py CODE HERE)
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from neurosymbolic_iot.data_processing.splits import split_train_val_test
from neurosymbolic_iot.utils.timer import timed

log = logging.getLogger(__name__)

DEFAULT_TIME_CANDS = ["timestamp", "time", "t", "ts", "datetime", "date_time", "start_time", "Time"]


def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="datetime64[ns, UTC]")

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        median = np.nanmedian(x.values) if np.isfinite(x.values).any() else np.nan
        if np.isfinite(median) and median > 1e12:
            return pd.to_datetime(x, errors="coerce", unit="ms", utc=True)
        if np.isfinite(median) and median > 1e9:
            return pd.to_datetime(x, errors="coerce", unit="s", utc=True)
        base = pd.Timestamp("1970-01-01", tz="UTC")
        return base + pd.to_timedelta(x, unit="s")

    dt = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", utc=True)
    if dt.notna().mean() >= 0.5:
        return dt

    td = pd.to_timedelta(s.astype(str).str.strip(), errors="coerce")
    if td.notna().mean() >= 0.5:
        base = pd.Timestamp("1970-01-01", tz="UTC")
        return base + td

    return dt


def _choose_existing_file(raw_dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = raw_dir / n
        if p.exists() and p.is_file():
            return p
    return None


def _load_label_matrix(cfg: Dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ds = cfg["datasets"]["sphere"]
    raw_dir = Path(ds["raw_dir"])

    label_fp = _choose_existing_file(raw_dir, ds.get("label_file_preference", ["activity.csv"]))
    if label_fp is None:
        raise RuntimeError("SPHERE: activity.csv not found (label matrix expected).")

    df = pd.read_csv(label_fp)
    cols = list(df.columns)

    tcol = _pick_col(cols, ds.get("time_column_candidates", DEFAULT_TIME_CANDS)) or ("t" if "t" in cols else None)
    if tcol is None:
        raise RuntimeError(f"SPHERE: could not detect time column in {label_fp.name}. Columns={cols}")

    label_cols = [c for c in cols if c != tcol and pd.api.types.is_numeric_dtype(df[c])]
    if len(label_cols) < 2:
        raise RuntimeError(f"SPHERE: not enough numeric label columns in {label_fp.name}. Columns={cols}")

    out = df[[tcol] + label_cols].copy()
    out = out.rename(columns={tcol: "timestamp"})
    out["timestamp"] = _parse_timestamp_series(out["timestamp"])

    scores = out[label_cols].to_numpy(dtype=float, copy=False)
    best_idx = np.nanargmax(scores, axis=1)
    best_val = scores[np.arange(scores.shape[0]), best_idx]

    out["label"] = np.array(label_cols, dtype=object)[best_idx]
    out = out.dropna(subset=["timestamp"]).copy()
    out = out[np.isfinite(best_val) & (best_val > 0)].copy()

    labels = out[["timestamp", "label"]].sort_values("timestamp").reset_index(drop=True)

    meta = {
        "label_file": label_fp.name,
        "time_col": tcol,
        "label_cols_count": int(len(label_cols)),
        "label_cols_sample": label_cols[:15],
        "n_rows": int(len(labels)),
        "n_labels": int(labels["label"].nunique()),
        "min_time": str(labels["timestamp"].min()),
        "max_time": str(labels["timestamp"].max()),
    }
    return labels, meta


def _load_stream(cfg: Dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ds = cfg["datasets"]["sphere"]
    raw_dir = Path(ds["raw_dir"])

    stream_fp = _choose_existing_file(
        raw_dir,
        ds.get("stream_file_preference", ["acceleration_corrected.csv", "acceleration.csv", "accel.csv", "pir.csv"]),
    )
    if stream_fp is None:
        raise RuntimeError("SPHERE: could not find stream file (acceleration*/accel/pir).")

    df = pd.read_csv(stream_fp)
    cols = list(df.columns)

    tcol = _pick_col(cols, ds.get("time_column_candidates", DEFAULT_TIME_CANDS)) or cols[0]
    df = df.rename(columns={tcol: "timestamp"})
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).copy()

    feat_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        df["dummy_feature"] = 1.0
        feat_cols = ["dummy_feature"]

    df = df[["timestamp"] + feat_cols].sort_values("timestamp").reset_index(drop=True)

    meta = {
        "stream_file": stream_fp.name,
        "time_col": tcol,
        "n_rows_stream": int(len(df)),
        "n_features": int(len(feat_cols)),
        "feature_cols_sample": feat_cols[:15],
        "min_time": str(df["timestamp"].min()),
        "max_time": str(df["timestamp"].max()),
    }
    return df, meta


def _attach_labels_nearest(stream: pd.DataFrame, labels: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    merged = pd.merge_asof(
        stream.sort_values("timestamp"),
        labels.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    merged = merged.dropna(subset=["label"]).reset_index(drop=True)
    return merged


def _make_time_windows(df: pd.DataFrame, window_seconds: int, stride_seconds: int, min_rows: int) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ["timestamp", "label"] and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        df["dummy_feature"] = 1.0
        feat_cols = ["dummy_feature"]

    start = df["timestamp"].min()
    end = df["timestamp"].max()
    win = pd.Timedelta(seconds=int(window_seconds))
    stride = pd.Timedelta(seconds=int(stride_seconds))

    out = []
    window_id = 0
    t = start
    while t + win <= end:
        wdf = df[(df["timestamp"] >= t) & (df["timestamp"] < t + win)]
        if len(wdf) >= min_rows:
            feats = wdf[feat_cols].mean().to_dict()
            mode = wdf["label"].mode(dropna=True)
            label = mode.iloc[0] if not mode.empty else None

            row = {"window_id": window_id, "start_time": t, "end_time": t + win, "n_rows": int(len(wdf)), "label": label}
            row.update(feats)
            out.append(row)
            window_id += 1
        t = t + stride

    return pd.DataFrame(out)


def preprocess_sphere(cfg: Dict) -> Dict[str, object]:
    ds = cfg["datasets"]["sphere"]
    out_windows = Path(cfg["output"]["sphere_windows"])
    out_meta = Path(cfg["output"]["sphere_meta"])

    window_seconds = int(ds.get("window_seconds", 30))
    stride_seconds = int(ds.get("stride_seconds", 15))
    min_rows = int(ds.get("min_rows_per_window", 5))
    tolerance_s = float(ds.get("label_join_tolerance_seconds", 5.0))

    timings: Dict[str, float] = {}

    with timed("load_labels", timings):
        labels, labels_meta = _load_label_matrix(cfg)

    with timed("load_stream", timings):
        stream, stream_meta = _load_stream(cfg)

    with timed("label_join", timings):
        labeled = _attach_labels_nearest(stream, labels, tolerance_s=tolerance_s)

    if labeled.empty or len(labeled) < 100:
        raise RuntimeError(
            "SPHERE: too few labeled rows after join. Increase label_join_tolerance_seconds "
            "or pick another stream_file_preference with matching time scale."
        )

    with timed("windowing", timings):
        windows = _make_time_windows(labeled, window_seconds, stride_seconds, min_rows)

    if windows.empty or len(windows) < 10:
        raise RuntimeError("SPHERE: too few windows; reduce min_rows_per_window or adjust window/stride.")

    with timed("splitting", timings):
        splits = split_train_val_test(
            windows,
            train=float(ds.get("train_ratio", 0.7)),
            val=float(ds.get("val_ratio", 0.15)),
            test=float(ds.get("test_ratio", 0.15)),
            seed=int(cfg.get("project", {}).get("seed", 42)),
            stratify_col="label",
        )
        tagged = []
        for split_name, sdf in splits.items():
            sdf = sdf.copy()
            sdf["split"] = split_name
            tagged.append(sdf)
        all_windows = pd.concat(tagged, ignore_index=True)

    with timed("persist", timings):
        out_windows.parent.mkdir(parents=True, exist_ok=True)
        all_windows.to_parquet(out_windows, index=False)

        meta = {
            "dataset": "SPHERE",
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "min_rows_per_window": min_rows,
            "label_join_tolerance_seconds": tolerance_s,
            "n_labeled_rows": int(len(labeled)),
            "n_windows": int(all_windows["window_id"].nunique()),
            "splits": {k: int(len(v)) for k, v in splits.items()},
            "labels_meta": labels_meta,
            "stream_meta": stream_meta,
            "timings_sec": timings,
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("SPHERE preprocessing complete: %s", out_windows.as_posix())
    return {"windows_path": str(out_windows), "meta_path": str(out_meta), "timings": timings}
