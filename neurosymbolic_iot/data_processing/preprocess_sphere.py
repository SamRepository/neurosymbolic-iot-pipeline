from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from neurosymbolic_iot.data_processing.splits import split_train_val_test
from neurosymbolic_iot.utils.timer import timed


log = logging.getLogger(__name__)


def _find_files(root: Path, globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for g in globs:
        files.extend(root.glob(g))
    return sorted(set([f for f in files if f.is_file()]))


def _guess_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_sphere_table(cfg: Dict) -> pd.DataFrame:
    ds = cfg["datasets"]["sphere"]
    raw_dir = Path(ds["raw_dir"])
    files = _find_files(raw_dir, ds.get("file_globs", ["**/*.csv"]))

    if not files:
        raise FileNotFoundError(f"No SPHERE files found under: {raw_dir}")

    parts: List[pd.DataFrame] = []
    for fp in tqdm(files, desc="Loading SPHERE files"):
        if fp.suffix.lower() == ".tsv":
            df = pd.read_csv(fp, sep="\t")
        else:
            df = pd.read_csv(fp)
        df["__source_file"] = fp.name
        parts.append(df)

    df_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df_all


def window_sphere_rows(
    df: pd.DataFrame,
    time_col: str,
    label_col: Optional[str],
    window_seconds: int,
    stride_seconds: int,
    min_rows: int = 1,
) -> pd.DataFrame:
    """Simple time-based windowing for SPHERE-like tables.

    This skeleton assumes a *flat table* with a timestamp column. Real SPHERE
    multimodal alignment can be plugged in later without changing downstream APIs.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    start = df[time_col].min()
    end = df[time_col].max()

    w = pd.Timedelta(seconds=window_seconds)
    s = pd.Timedelta(seconds=stride_seconds)

    rows = []
    window_id = 0
    cur = start
    feature_cols = [c for c in df.columns if c not in {time_col, label_col, "__source_file"}]

    while cur <= end:
        cur_end = cur + w
        wdf = df[(df[time_col] >= cur) & (df[time_col] < cur_end)]
        if len(wdf) >= min_rows:
            # Aggregation strategy: mean for numeric columns; mode for non-numeric
            feats = {}
            for c in feature_cols:
                if pd.api.types.is_numeric_dtype(wdf[c]):
                    feats[c] = float(wdf[c].mean())
                else:
                    vc = wdf[c].astype(str).value_counts()
                    feats[c] = vc.idxmax() if not vc.empty else None

            label = None
            if label_col and label_col in wdf.columns:
                vc = wdf[label_col].dropna().astype(str).value_counts()
                label = vc.idxmax() if not vc.empty else None

            rows.append(
                {
                    "window_id": window_id,
                    "start_time": cur,
                    "end_time": cur_end,
                    "n_rows": int(len(wdf)),
                    "label": label,
                    **feats,
                }
            )
            window_id += 1
        cur = cur + s

    return pd.DataFrame(rows)


def preprocess_sphere(cfg: Dict) -> Dict[str, object]:
    ds = cfg["datasets"]["sphere"]
    out_windows = Path(cfg["output"]["sphere_windows"])
    out_meta = Path(cfg["output"]["sphere_meta"])

    timings: Dict[str, float] = {}
    with timed("load", timings):
        df = load_sphere_table(cfg)

    time_col = _guess_column(df, ds.get("time_column_candidates", ["timestamp"]))
    if not time_col:
        raise ValueError(
            "Could not find a timestamp column in SPHERE data. "
            f"Candidates tried: {ds.get('time_column_candidates')}. "
            f"Columns found: {list(df.columns)[:30]} ..."
        )

    label_col = _guess_column(df, ds.get("label_column_candidates", ["label", "activity"]))

    with timed("windowing", timings):
        windows = window_sphere_rows(
            df=df,
            time_col=time_col,
            label_col=label_col,
            window_seconds=int(ds["window_seconds"]),
            stride_seconds=int(ds["stride_seconds"]),
            min_rows=int(ds.get("min_rows_per_window", 1)),
        )

    if windows.empty:
        raise RuntimeError("No SPHERE windows created. Check window params / timestamp parsing.")

    # Split train/val/test at window level (Phase 1 baseline)
    split_cfg = ds.get("split", {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42, "stratify": True})
    stratify_col = "label" if split_cfg.get("stratify", True) else None
    with timed("split", timings):
        splits = split_train_val_test(
            windows,
            train=float(split_cfg["train"]),
            val=float(split_cfg["val"]),
            test=float(split_cfg["test"]),
            seed=int(split_cfg.get("seed", 42)),
            stratify_col=stratify_col,
        )

    with timed("persist", timings):
        out_windows.parent.mkdir(parents=True, exist_ok=True)
        # Persist all windows and provide split indices via meta (simple, portable)
        windows.to_parquet(out_windows, index=False)

        meta = {
            "dataset": "SPHERE",
            "n_raw_rows": int(len(df)),
            "n_windows": int(len(windows)),
            "time_col": time_col,
            "label_col": label_col,
            "window_seconds": int(ds["window_seconds"]),
            "stride_seconds": int(ds["stride_seconds"]),
            "splits": {k: int(len(v)) for k, v in splits.items()},
            "timings_sec": timings,
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("SPHERE preprocessing complete: %s", out_windows.as_posix())
    return {"windows_path": str(out_windows), "meta_path": str(out_meta), "timings": timings}
