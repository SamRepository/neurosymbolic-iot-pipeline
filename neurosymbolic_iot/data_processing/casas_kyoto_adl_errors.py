from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


# From Zenodo record 10.5281/zenodo.15712834 (tasks + injected errors). :contentReference[oaicite:1]{index=1}
TASKS: Dict[int, str] = {
    1: "PhoneCall",
    2: "WashHands",
    3: "Cook",
    4: "Eat",
    5: "Clean",
}

ERRORS: Dict[int, str] = {
    1: "Dials wrong number then redials",
    2: "Does not turn water off after washing hands",
    3: "Does not turn burner off after cooking oatmeal",
    4: "Does not bring medicine container to dining room",
    5: "Does not use water to clean dishes",
}

FILE_RE = re.compile(r"^(p\d+)\.t(\d+)\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedName:
    participant: str
    task_id: int
    task_name: str


def parse_filename(fp: Path) -> Optional[ParsedName]:
    m = FILE_RE.match(fp.name)
    if not m:
        return None
    participant = m.group(1).lower()
    task_id = int(m.group(2))
    task_name = TASKS.get(task_id, f"Task{task_id}")
    return ParsedName(participant=participant, task_id=task_id, task_name=task_name)


def detect_split_and_error(fp: Path) -> Tuple[str, bool]:
    """
    We expect files to be under folders named like:
      .../adl_error/p01.t1.csv
      .../adl_noerror/p01.t1.csv
    """
    parts = [p.lower() for p in fp.parts]
    if "adl_error" in parts:
        return ("adl_error", True)
    if "adl_noerror" in parts:
        return ("adl_noerror", False)
    # Fallback: infer from parent folder name
    pname = fp.parent.name.lower()
    if "error" in pname and "noerror" not in pname:
        return ("adl_error", True)
    if "noerror" in pname:
        return ("adl_noerror", False)
    return ("unknown", False)


def _is_header_row(row: List[str]) -> bool:
    joined = ",".join(row).lower()
    return any(k in joined for k in ["date", "time", "sensor", "message", "value", "activity", "label", "timestamp"])


def load_kyoto_adl_errors_events(raw_dir: Path, file_globs: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads CASAS scripted ADL tasks with/without injected errors:
    - One file per participant per task (e.g., p01.t1.csv)
    - Each row: date, time, sensor, message  (Zenodo description). :contentReference[oaicite:2]{index=2}
    Returns a canonical events DataFrame with:
      timestamp, sensor, value, activity, participant, task_id, task_name, has_error, error_desc, dataset_split, source_file
    """
    if file_globs is None:
        file_globs = ["**/*.csv"]

    files: List[Path] = []
    for g in file_globs:
        files.extend(list(raw_dir.glob(g)))
    files = sorted([f for f in set(files) if f.is_file()])

    if not files:
        raise FileNotFoundError(f"No CASAS Kyoto ADL error/noerror CSV files found under: {raw_dir}")

    rows_out: List[Dict[str, object]] = []

    for fp in files:
        parsed = parse_filename(fp)
        if parsed is None:
            continue

        dataset_split, has_error = detect_split_and_error(fp)
        error_desc = ERRORS.get(parsed.task_id) if has_error else None

        with fp.open("r", encoding="utf-8", errors="ignore", newline="") as f:
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

                # Expected: date, time, sensor, message
                if len(row) < 4:
                    continue

                date = row[0]
                time = row[1]
                sensor = row[2]
                value = row[3]

                rows_out.append(
                    {
                        "timestamp": f"{date} {time}",
                        "sensor": sensor,
                        "value": value,
                        "activity": parsed.task_name,         # clean task label
                        "participant": parsed.participant,
                        "task_id": parsed.task_id,
                        "task_name": parsed.task_name,
                        "has_error": bool(has_error),
                        "error_desc": error_desc,
                        "dataset_split": dataset_split,        # adl_error / adl_noerror
                        "source_file": fp.name,
                    }
                )

    df = pd.DataFrame(rows_out)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "sensor",
                "value",
                "activity",
                "participant",
                "task_id",
                "task_name",
                "has_error",
                "error_desc",
                "dataset_split",
                "source_file",
            ]
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "sensor", "value", "activity"]).reset_index(drop=True)
    df = df.sort_values(["participant", "timestamp"]).reset_index(drop=True)

    log.info(
        "Loaded CASAS Kyoto ADL (errors) events: n=%d participants=%d tasks=%d",
        len(df),
        df["participant"].nunique(),
        df["task_id"].nunique(),
    )
    return df
