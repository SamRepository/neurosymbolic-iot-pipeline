from __future__ import annotations

import csv
import logging
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from neurosymbolic_iot.data_processing.windowing import rolling_event_windows
from neurosymbolic_iot.data_processing.casas_kyoto_adl_errors import load_kyoto_adl_errors_events
from neurosymbolic_iot.utils.timer import timed

log = logging.getLogger(__name__)


def _find_files(root: Path, globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for g in globs:
        files.extend(root.glob(g))
    return sorted(set([f for f in files if f.is_file()]))

# updated for casas preprocessing

def _is_header_row(row: List[str]) -> bool:
    joined = ",".join(row).lower()
    return any(k in joined for k in ["date", "time", "timestamp", "sensor", "value", "activity", "label"])


def _parse_casas_ragged_csv(path: Path) -> pd.DataFrame:
    """
    CASAS CSV logs can be ragged (variable number of columns per row).
    Parse line-by-line:
      date, time, sensor, value, ...extras...
    Keep:
      - timestamp = date + " " + time
      - sensor
      - value
      - activity = " ".join(extras)  (stores any additional columns safely)
    """
    out: List[Dict[str, Optional[str]]] = []

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]
            if first and _is_header_row(row):
                first = False
                continue
            first = False

            if len(row) < 4:
                continue

            date = row[0]
            time = row[1]
            sensor = row[2]
            value = row[3]
            extras = row[4:] if len(row) > 4 else []
            activity = " ".join([e for e in extras if e]) if extras else None

            out.append(
                {
                    "timestamp": f"{date} {time}",
                    "sensor": sensor,
                    "value": value,
                    "activity": activity,
                }
            )

    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "sensor", "value", "activity"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "sensor", "value"]).reset_index(drop=True)
    return df


def _parse_casas_text_like(path: Path) -> pd.DataFrame:
    """
    Space-delimited CASAS logs fallback:
      YYYY-MM-DD HH:MM:SS.sss SENSOR VALUE [LABEL...]
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts = " ".join(parts[0:2])
            sensor = parts[2]
            value = parts[3]
            activity = " ".join(parts[4:]) if len(parts) > 4 else None
            rows.append((ts, sensor, value, activity))

    df = pd.DataFrame(rows, columns=["timestamp", "sensor", "value", "activity"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "sensor", "value"]).reset_index(drop=True)
    return df


def load_casas_events(cfg: Dict) -> pd.DataFrame:
    ds = cfg["datasets"]["casas"]
    raw_dir = Path(ds["raw_dir"])
 #   files = _find_files(raw_dir, ds.get("file_globs", ["**/*.csv", "**/*.txt", "**/*.log"]))
    casas_format = str(ds.get("format", "auto")).lower().strip()
    if casas_format == "kyoto_adl_errors":
        # Zenodo 10.5281/zenodo.15712834: pXX.tY.csv files inside adl_error/ and adl_noerror/ :contentReference[oaicite:3]{index=3}
        events = load_kyoto_adl_errors_events(
            raw_dir=raw_dir,
            file_globs=ds.get("file_globs", ["**/*.csv"]),
        )
        # Ensure expected columns exist
        if "activity" not in events.columns:
            events["activity"] = None
        return events

    files = _find_files(raw_dir, ds.get("file_globs", ["**/*.csv", "**/*.txt", "**/*.log"]))
 
    if not files:
        raise FileNotFoundError(f"No CASAS files found under: {raw_dir}")

    parts: List[pd.DataFrame] = []
    for fp in tqdm(files, desc="Loading CASAS files"):
        if fp.suffix.lower() == ".csv":
            df = _parse_casas_ragged_csv(fp)
        else:
            df = _parse_casas_text_like(fp)

        if not df.empty:
            parts.append(df)

    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["timestamp", "sensor", "value", "activity"])
    events = events.dropna(subset=["timestamp", "sensor", "value"]).reset_index(drop=True)
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

    with timed("persist", timings):
        if X.empty:
            raise RuntimeError("No CASAS windows created. Check window params / parsing.")

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


