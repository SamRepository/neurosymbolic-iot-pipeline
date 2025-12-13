from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from neurosymbolic_iot.data_processing.splits import split_train_val_test
from neurosymbolic_iot.utils.timer import timed

log = logging.getLogger(__name__)


DEFAULT_TIME_CANDS = ["timestamp", "time", "t", "ts", "datetime", "date_time", "start_time", "start", "Time"]
DEFAULT_LABEL_CANDS = ["activity", "label", "class", "Activity", "Label"]


def _find_files(root: Path, globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for g in globs:
        files.extend(root.glob(g))
    return sorted(set([f for f in files if f.is_file()]))


def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    # exact match first
    for c in candidates:
        if c in columns:
            return c
    # case-insensitive match
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    """
    Robust timestamp parsing:
    - string datetimes
    - numeric epoch seconds/ms
    - HH:MM:SS(.fff) time-only -> treated as timedelta from origin
    """
    if s is None:
        return pd.Series(dtype="datetime64[ns, UTC]")

    # If numeric -> epoch
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        # Heuristic: ms if very large
        median = np.nanmedian(x.values) if np.isfinite(x.values).any() else np.nan
        if np.isfinite(median) and median > 1e12:
            return pd.to_datetime(x, errors="coerce", unit="ms", utc=True)
        if np.isfinite(median) and median > 1e9:
            return pd.to_datetime(x, errors="coerce", unit="s", utc=True)
        # otherwise seconds offset
        base = pd.Timestamp("1970-01-01", tz="UTC")
        td = pd.to_timedelta(x, unit="s")
        return base + td

    # Try normal datetime parsing
    dt = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", utc=True)
    if dt.notna().mean() >= 0.5:
        return dt

    # Try time-only parsing (HH:MM:SS)
    td = pd.to_timedelta(s.astype(str).str.strip(), errors="coerce")
    if td.notna().mean() >= 0.5:
        base = pd.Timestamp("1970-01-01", tz="UTC")
        return base + td

    return dt  # whatever we got


def _load_one_csv(fp: Path) -> pd.DataFrame:
    # SPHERE CSVs are standard; keep this simple
    return pd.read_csv(fp)


def load_sphere_labeled_table(cfg: Dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ds = cfg["datasets"]["sphere"]
    raw_dir = Path(ds["raw_dir"])

    globs = ds.get("file_globs", ["*.csv", "**/*.csv"])
    files = _find_files(raw_dir, globs)
    if not files:
        raise FileNotFoundError(f"No SPHERE files found under: {raw_dir}")

    time_cands = ds.get("time_column_candidates", DEFAULT_TIME_CANDS)
    label_cands = ds.get("label_column_candidates", DEFAULT_LABEL_CANDS)

    kept = []
    kept_info = []

    for fp in tqdm(files, desc="Loading SPHERE files"):
        try:
            # Read a small sample to detect columns quickly
            sample = pd.read_csv(fp, nrows=5)
            cols = list(sample.columns)

            tcol = _pick_col(cols, time_cands)
            lcol = _pick_col(cols, label_cands)

            # Phase 1 baseline: only keep files that have BOTH time and label
            if not tcol or not lcol:
                continue

            df = _load_one_csv(fp)
            df = df.rename(columns={tcol: "timestamp", lcol: "label"})

            df["timestamp"] = _parse_timestamp_series(df["timestamp"])
            df = df.dropna(subset=["timestamp", "label"]).copy()
            df["label"] = df["label"].astype(str).str.strip()

            # Keep numeric columns as features
            feature_cols = [c for c in df.columns if c not in ["timestamp", "label"] and pd.api.types.is_numeric_dtype(df[c])]
            # If none numeric, create a dummy feature so windowing still works
            if not feature_cols:
                df["dummy_feature"] = 1.0
                feature_cols = ["dummy_feature"]

            df = df[["timestamp", "label"] + feature_cols].copy()
            kept.append(df)

            kept_info.append(
                {
                    "file": fp.name,
                    "rows_after_clean": int(len(df)),
                    "time_col": tcol,
                    "label_col": lcol,
                    "n_features": int(len(feature_cols)),
                }
            )
        except Exception as e:
            log.warning("Failed reading %s: %s", fp.name, e)

    if not kept:
        raise RuntimeError(
            "No SPHERE CSV contained BOTH a timestamp column and an activity label column. "
            "Adjust datasets.sphere.file_globs and *_column_candidates in config/base.yaml."
        )

    data = pd.concat(kept, ignore_index=True)
    data = data.sort_values("timestamp").reset_index(drop=True)

    meta = {
        "kept_files": kept_info,
        "n_rows_total": int(len(data)),
        "n_labels": int(data["label"].nunique()),
        "min_time": str(data["timestamp"].min()),
        "max_time": str(data["timestamp"].max()),
    }
    return data, meta


def make_time_windows(
    df: pd.DataFrame,
    window_seconds: int,
    stride_seconds: int,
    min_rows_per_window: int = 5,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c not in ["timestamp", "label"] and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        df["dummy_feature"] = 1.0
        numeric_cols = ["dummy_feature"]

    start = df["timestamp"].min()
    end = df["timestamp"].max()
    win = pd.Timedelta(seconds=int(window_seconds))
    stride = pd.Timedelta(seconds=int(stride_seconds))

    out_rows = []
    window_id = 0
    t = start

    while t + win <= end:
        wdf = df[(df["timestamp"] >= t) & (df["timestamp"] < t + win)]
        if len(wdf) >= min_rows_per_window:
            feats = wdf[numeric_cols].mean().to_dict()
            label = wdf["label"].mode(dropna=True)
            label = label.iloc[0] if not label.empty else None

            row = {
                "window_id": window_id,
                "start_time": t,
                "end_time": t + win,
                "n_rows": int(len(wdf)),
                "label": label,
            }
            row.update(feats)
            out_rows.append(row)
            window_id += 1

        t = t + stride

    return pd.DataFrame(out_rows)


def preprocess_sphere(cfg: Dict) -> Dict[str, object]:
    ds = cfg["datasets"]["sphere"]

    out_windows = Path(cfg["output"]["sphere_windows"])
    out_meta = Path(cfg["output"]["sphere_meta"])

    window_seconds = int(ds.get("window_seconds", 30))
    stride_seconds = int(ds.get("stride_seconds", 15))
    min_rows_per_window = int(ds.get("min_rows_per_window", 5))

    timings: Dict[str, float] = {}
    with timed("load", timings):
        data, load_meta = load_sphere_labeled_table(cfg)

    with timed("windowing", timings):
        windows = make_time_windows(
            data,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_rows_per_window=min_rows_per_window,
        )

    if windows.empty or len(windows) < 3:
        raise RuntimeError(
            f"SPHERE produced too few windows (n_windows={len(windows)}). "
            "This typically means timestamp parsing failed or the chosen input file has too little labeled data. "
            "Try restricting file_globs to activity.csv or per_ann_activity_*.csv."
        )

    with timed("splitting", timings):
        splits = split_train_val_test(
            windows,
            train=float(ds.get("train_ratio", 0.7)),
            val=float(ds.get("val_ratio", 0.15)),
            test=float(ds.get("test_ratio", 0.15)),
            seed=int(cfg.get("project", {}).get("seed", 42)),
            stratify_col="label",
        )

        # Add split column
        windows_split = []
        for split_name, sdf in splits.items():
            sdf = sdf.copy()
            sdf["split"] = split_name
            windows_split.append(sdf)
        all_windows = pd.concat(windows_split, ignore_index=True)

    with timed("persist", timings):
        out_windows.parent.mkdir(parents=True, exist_ok=True)
        all_windows.to_parquet(out_windows, index=False)

        meta = {
            "dataset": "SPHERE",
            "n_rows_input": int(load_meta["n_rows_total"]),
            "n_windows": int(all_windows["window_id"].nunique()),
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "min_rows_per_window": min_rows_per_window,
            "splits": {k: int(len(v)) for k, v in splits.items()},
            "load_details": load_meta,
            "timings_sec": timings,
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("SPHERE preprocessing complete: %s", out_windows.as_posix())
    return {"windows_path": str(out_windows), "meta_path": str(out_meta), "timings": timings}
