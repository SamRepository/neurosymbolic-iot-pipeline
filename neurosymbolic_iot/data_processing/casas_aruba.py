"""
CASAS Aruba / Zenodo-15708568 labeled-dataset loader.

This loader handles the **CASAS Smart Home dataset** published on Zenodo
under DOI `10.5281/zenodo.15708568` — the labeled subset (236 MB). The
release bundles 82 home recordings across the ``hh``, ``ihs``, ``mn``,
``mv``, ``mva``, ``rw``, ``tm`` testbed series, each with human-annotated
activity labels.

File layout expected on disk
----------------------------
After extracting ``labeled_data.zip``, the canonical project layout is::

    data/raw/casas_aruba/labeled/
        hh101.csv
        hh102.csv
        ...
        tm005.csv

Each CSV file is one home, one event per line, with the schema::

    YYYY-MM-DD, HH:MM:SS.ffffff, SENSOR, STATE [, ACTIVITY="begin|end"]

Example (from ``hh101.csv``)::

    2012-07-20,10:38:54.512364,OutsideDoor,ON,Step_Out="begin"
    2012-07-20,10:50:54.933393,OutsideDoor,OFF,Step_Out="end"
    2012-07-20,11:09:18.9523,Bathroom,ON,Toilet="begin"

Important format notes
----------------------
* Sensor names are ROOM-LEVEL (Bathroom, Bedroom, Kitchen, LivingRoom,
  DiningRoom, OutsideDoor). No separate sensor→room mapping is needed;
  the loader emits the sensor string both as the canonical
  ``sensor`` column and as a ``room`` hint.
* Activity boundary markers use the syntax ``Activity="begin"`` /
  ``Activity="end"`` embedded as the fifth field. Everything between a
  matched begin/end pair is assigned that activity.
* The release uses fine-grained activities (34 distinct labels per
  ``hh101.csv``: ``Cook_Breakfast``, ``Cook_Lunch``, ``Eat_Dinner`` …).
  This loader returns those labels verbatim; downstream consumers can
  collapse them via ``ACTIVITY_CANONICAL_MAP`` if a coarser vocabulary
  is preferred.

Both the *classic* Aruba (``aruba.data``, tab/space-separated) and this
Zenodo CSV format are accepted — the loader auto-detects per file by
extension and delimiter.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical activity vocabulary
# ---------------------------------------------------------------------------
# The Zenodo labeled bundle uses fine-grained activities. The map below
# collapses them into the AAL vocabulary already targeted by the SWRL rule
# programme (rules 3-11 in symbolic_reasoner / rule_executor).
# Activities not in this map fall through with their original label.

ACTIVITY_CANONICAL_MAP: Dict[str, str] = {
    # Cooking family
    "Cook": "MealPreparation",
    "Cook_Breakfast": "MealPreparation",
    "Cook_Lunch": "MealPreparation",
    "Cook_Dinner": "MealPreparation",
    # Eating family
    "Eat": "Eating",
    "Eat_Breakfast": "Eating",
    "Eat_Lunch": "Eating",
    "Eat_Dinner": "Eating",
    "Drink": "Eating",
    # Hygiene family
    "Bathe": "PersonalHygiene",
    "Groom": "PersonalHygiene",
    "Dress": "PersonalHygiene",
    "Personal_Hygiene": "PersonalHygiene",
    "Toilet": "PersonalHygiene",
    "Bed_Toilet_Transition": "PersonalHygiene",
    # Sleep family
    "Sleep": "Sleeping",
    "Go_To_Sleep": "Sleeping",
    "Wake_Up": "Sleeping",
    "Sleep_Out_Of_Bed": "Sleeping",
    # Housework family
    "Wash_Dishes": "Housekeeping",
    "Wash_Breakfast_Dishes": "Housekeeping",
    "Wash_Lunch_Dishes": "Housekeeping",
    "Wash_Dinner_Dishes": "Housekeeping",
    # Medication, entertainment, etc. mapped to broader classes
    "Evening_Meds": "Medication",
    "Morning_Meds": "Medication",
    "Watch_TV": "Relax",
    "Read": "Relax",
    "Relax": "Relax",
    "Entertain_Guests": "Relax",
    "Phone": "PhoneCall",
    "Work_At_Table": "Work",
    # Transitions kept distinct
    "Enter_Home": "EnterHome",
    "Leave_Home": "LeaveHome",
    "Step_Out": "LeaveHome",
}

# The canonical activity set that survives the map above.
ARUBA_ACTIVITIES = sorted(set(ACTIVITY_CANONICAL_MAP.values()))

# Activity="begin|end" boundary regex on the trailing field of a CSV line.
_BOUNDARY_RE = re.compile(r'([A-Za-z_]+)\s*=\s*"(begin|end)"', re.IGNORECASE)

# Classic Aruba line whitespace splitter (when format == .data / .txt).
_WS_SPLIT_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Zenodo CSV format
# ---------------------------------------------------------------------------

def _canonical_activity(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return ACTIVITY_CANONICAL_MAP.get(raw, raw)


def _parse_zenodo_csv(path: Path) -> List[Dict[str, object]]:
    """Parse one Zenodo-labeled CASAS home CSV.

    Returns one event dict per line. Activity labels are filled in
    between matched ``Activity="begin"`` / ``Activity="end"`` pairs.
    """
    rows: List[Dict[str, object]] = []
    open_spans: List[tuple[str, int]] = []  # (raw_activity, begin_idx)

    home = path.stem  # e.g. 'hh101'

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        for line_num, parts in enumerate(reader, start=1):
            parts = [p.strip() for p in parts]
            if len(parts) < 4:
                continue
            date, time, sensor, value = parts[0], parts[1], parts[2], parts[3]
            ts = f"{date} {time}"

            row_idx = len(rows)
            rows.append({
                "timestamp": ts,
                "sensor": sensor,
                "value": value,
                "activity": None,           # filled by begin/end propagation
                "participant": home,
                "task_id": 0,
                "task_name": "free_living",
                "has_error": False,
                "error_desc": None,
                "dataset_split": "zenodo_labeled",
                "source_file": path.name,
            })

            # Inspect remaining fields for an Activity="begin|end" marker.
            tail = ",".join(parts[4:])
            if not tail:
                continue
            m = _BOUNDARY_RE.search(tail)
            if not m:
                continue
            raw_activity, status = m.group(1), m.group(2).lower()
            if status == "begin":
                open_spans.append((raw_activity, row_idx))
            else:  # end
                # Match the most recent open span with this label.
                match_pos: Optional[int] = None
                for i in range(len(open_spans) - 1, -1, -1):
                    if open_spans[i][0] == raw_activity:
                        match_pos = i
                        break
                if match_pos is None:
                    log.debug("%s line %d: 'end %s' with no matching 'begin' — dropped",
                              path.name, line_num, raw_activity)
                    continue
                _, begin_idx = open_spans.pop(match_pos)
                canonical = _canonical_activity(raw_activity)
                for k in range(begin_idx, row_idx + 1):
                    rows[k]["activity"] = canonical

    if open_spans:
        log.warning("%s: %d activity span(s) never closed", path.name, len(open_spans))
    return rows


# ---------------------------------------------------------------------------
# Classic Aruba .data format (tab/space-separated) — preserved for users
# who downloaded the original CASAS release rather than the Zenodo bundle.
# ---------------------------------------------------------------------------

def _parse_classic_aruba(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    open_spans: List[tuple[str, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _WS_SPLIT_RE.split(line)
            if len(parts) < 4:
                continue
            date, time, sensor, value = parts[0], parts[1], parts[2], parts[3]
            raw_activity = parts[4] if len(parts) >= 5 else None
            status = parts[5].lower() if len(parts) >= 6 else None

            row_idx = len(rows)
            rows.append({
                "timestamp": f"{date} {time}",
                "sensor": sensor,
                "value": value,
                "activity": None,
                "participant": "aruba_single",
                "task_id": 0,
                "task_name": "free_living",
                "has_error": False,
                "error_desc": None,
                "dataset_split": "aruba_classic",
                "source_file": path.name,
            })

            if raw_activity and status in ("begin", "end"):
                if status == "begin":
                    open_spans.append((raw_activity, row_idx))
                else:
                    match_pos: Optional[int] = None
                    for i in range(len(open_spans) - 1, -1, -1):
                        if open_spans[i][0] == raw_activity:
                            match_pos = i
                            break
                    if match_pos is None:
                        log.debug("%s line %d: 'end %s' without matching begin",
                                  path.name, line_num, raw_activity)
                        continue
                    _, begin_idx = open_spans.pop(match_pos)
                    canonical = _canonical_activity(raw_activity)
                    for k in range(begin_idx, row_idx + 1):
                        rows[k]["activity"] = canonical

    if open_spans:
        log.warning("%s: %d activity span(s) never closed", path.name, len(open_spans))
    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_CANONICAL_COLUMNS = [
    "timestamp", "sensor", "value", "activity", "participant",
    "task_id", "task_name", "has_error", "error_desc",
    "dataset_split", "source_file",
]


def load_aruba_events(
    raw_dir: Path,
    file_globs: Optional[List[str]] = None,
    homes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load CASAS Aruba / Zenodo-labeled events into the canonical schema.

    Parameters
    ----------
    raw_dir
        Directory containing either:
          * a single classic Aruba file (e.g. ``data``, ``aruba.data``), or
          * the extracted Zenodo labeled bundle laid out as
            ``raw_dir/labeled/{home_id}.csv``.
    file_globs
        Optional override of the default search patterns.
    homes
        Optional whitelist of home identifiers (CSV stem, e.g. ``hh101``).
        If set, only those homes are loaded. Useful for running a CV
        ablation on a single home for direct comparison with the Kyoto
        result, or for evaluating cross-home generalisation.

    Returns
    -------
    DataFrame with columns matching ``load_kyoto_adl_errors_events``.
    """
    if file_globs is None:
        file_globs = [
            # Zenodo labeled bundle
            "labeled/*.csv",
            "*.csv",
            # Classic Aruba release
            "data",
            "aruba.data",
            "aruba.txt",
            "**/aruba*",
        ]

    seen: set = set()
    files: List[Path] = []
    for pattern in file_globs:
        for hit in raw_dir.glob(pattern):
            if hit.is_file() and hit not in seen:
                seen.add(hit)
                files.append(hit)
    files = sorted(files)

    if homes:
        homeset = {h.lower() for h in homes}
        files = [f for f in files if f.stem.lower() in homeset]

    if not files:
        raise FileNotFoundError(
            f"No CASAS Aruba / Zenodo CASAS file found under {raw_dir}. "
            f"Tried globs={file_globs}."
        )

    all_rows: List[Dict[str, object]] = []
    csv_count = 0
    classic_count = 0
    for fp in files:
        log.info("Parsing %s", fp)
        if fp.suffix.lower() == ".csv":
            all_rows.extend(_parse_zenodo_csv(fp))
            csv_count += 1
        else:
            all_rows.extend(_parse_classic_aruba(fp))
            classic_count += 1

    if not all_rows:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)

    df = pd.DataFrame(all_rows)
    log.info(
        "Loaded CASAS events: n=%d (csv_files=%d, classic_files=%d) "
        "sensors=%d activities=%d annotated_rows=%d",
        len(df), csv_count, classic_count,
        df["sensor"].nunique(),
        df["activity"].dropna().nunique(),
        df["activity"].notna().sum(),
    )
    return df
