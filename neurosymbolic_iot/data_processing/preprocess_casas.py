from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from neurosymbolic_iot.data_processing.windowing import rolling_event_windows
from neurosymbolic_iot.utils.timer import timed


log = logging.getLogger(__name__)


def _find_files(root: Path, globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for g in globs:
        files.extend(root.glob(g))
    # de-dup + stable order
    return sorted(set([f for f in files if f.is_file()]))


def _parse_casas_text_like(path: Path, label_candidates: List[str]) -> pd.DataFrame:
    """Best-effort parser for CASAS-like event logs.

    Expected patterns often look like:
      YYYY-MM-DD HH:MM:SS.sss  SENSOR  VALUE  [LABEL]

    Since variants exist, we attempt:
      - whitespace split
      - fallback CSV parsing
    """
    # Try CSV first for .csv
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # timestamp may be in 2 tokens (date time)
            ts = " ".join(parts[0:2])
            sensor = parts[2]
            value = parts[3]
            activity = None
            if len(parts) > 4:
                # label may be last token(s); best-effort take the remainder
                activity = " ".join(parts[4:])
            rows.append((ts, sensor, value, activity))

    df = pd.DataFrame(rows, columns=["timestamp", "sensor", "value", "activity"])
    return df


def load_casas_events(cfg: Dict) -> pd.DataFrame:
    ds = cfg["datasets"]["casas"]
    raw_dir = Path(ds["raw_dir"])
    files = _find_files(raw_dir, ds.get("file_globs", ["**/*.txt"]))

    if not files:
        raise FileNotFoundError(f"No CASAS files found under: {raw_dir}")

    all_parts: List[pd.DataFrame] = []
    for fp in tqdm(files, desc="Loading CASAS files"):
        df = _parse_casas_text_like(fp, ds.get("label_column_candidates", []))
        all_parts.append(df)

    events = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    # Normalize timestamp
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    events = events.dropna(subset=["timestamp", "sensor", "value"]).reset_index(drop=True)

    # If activity column missing, create it
    if "activity" not in events.columns:
        events["activity"] = None

    return events


def preprocess_casas(cfg: Dict) -> Dict[str, object]:
    ds = cfg["datasets"]["casas"]
    out_windows = Path(cfg["output"]["casas_windows"])
    out_meta = Path(cfg["output"]["casas_meta"])

    timings: Dict[str, float] = {}
    with timed("load", timings):
        events = load_casas_events(cfg)

    with timed("windowing", timings):
        X, Y = rolling_event_windows(
            events=events,
            window_minutes=int(ds["window_minutes"]),
            stride_minutes=int(ds["stride_minutes"]),
            min_events=int(ds.get("min_events_per_window", 1)),
            label_mode=str(ds.get("label_mode", "majority")),
        )

    # Merge features + meta for a single artifact (convenient for later phases)
    with timed("persist", timings):
        if X.empty:
            raise RuntimeError("No CASAS windows created. Check window params / raw parsing.")

        df = Y.merge(X, on="window_id", how="left")
        out_windows.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_windows, index=False)

        meta = {
            "dataset": "CASAS",
            "n_events": int(len(events)),
            "n_windows": int(df["window_id"].nunique()),
            "window_minutes": int(ds["window_minutes"]),
            "stride_minutes": int(ds["stride_minutes"]),
            "feature_mode": ds.get("feature_mode"),
            "label_mode": ds.get("label_mode"),
            "timings_sec": timings,
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("CASAS preprocessing complete: %s", out_windows.as_posix())
    return {"windows_path": str(out_windows), "meta_path": str(out_meta), "timings": timings}
