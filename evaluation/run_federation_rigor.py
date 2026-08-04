"""
Federation Experiment Rigor (E1 - FGCS revision)
================================================
Addresses reviewer items R1-3, R1-4, R2-9: multi-trial variance for the
federation headline, per-category precision/recall/F1 against a programmatic
ground truth, injection-rate sensitivity, and an explicit reconciliation of
injected-anomaly counts vs total detections.

The legacy experiment (``evaluation/run_cross_dataset_federation.py``) is
imported, not modified: its generators and KG builders produce the data, and
its exact query semantics are preserved as the ``as_published`` metric path.
This script adds:

1. A ground-truth registry: every injected anomaly gets an id, type, timestamp
   and the indices of its CASAS/SPHERE prediction pair. Injection timestamps
   are randomized per trial (collision-checked) instead of the legacy fixed
   12:00/14:00/18:00 grid, so trials are genuinely randomized.
2. Generative labels for the background stream (low-confidence, night-time,
   multi-room motion) derived from the generator itself - the synthetic data
   makes ground truth programmatic, no human annotation involved.
3. Binding-returning queries plus *event-scoped* corrected variants of the two
   legacy metrics that were graph-static sensor-pair counts (multi-room motion,
   cross-source location conflict). Event scoping requires event->room
   provenance triples (``nsiot:involvesMotionInRoom``) which this script adds
   to its own graphs only; the shared KG loader is untouched.
4. Per-category TP/FP/FN scoring against the registry / generative labels.

Usage:
  PYTHONPATH=. python evaluation/run_federation_rigor.py --config config/base.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdflib import Graph
from scipy import stats as scistats

from evaluation.run_cross_dataset_federation import (
    NSIOT,
    _detect_anomalies_federated,
    _detect_anomalies_single,
    _generate_casas_predictions,
    _generate_sphere_predictions,
    build_federated_kg,
)
from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
    _is_night_time,
    _resolve_sensor_type,
    build_kg_from_predictions,
    load_sensor_map,
)

log = logging.getLogger(__name__)

NSIOT_STR = "http://example.org/neuro-symbolic-iot#"
TIME_STR = "http://www.w3.org/2006/time#"

BASE_DATE = datetime(2024, 6, 15)

# ---------------------------------------------------------------------------
# Injection (payloads identical to the legacy script; timestamps randomizable)
# ---------------------------------------------------------------------------

_INJECTION_ORDER = (
    "temporal_activity_contradiction",
    "spatial_motion_inconsistency",
    "activity_posture_mismatch",
)

_LEGACY_BASE_HOURS = {
    "temporal_activity_contradiction": 12,
    "spatial_motion_inconsistency": 14,
    "activity_posture_mismatch": 18,
}


def _injection_payloads(kind: str, ts_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """CASAS/SPHERE prediction payloads for one injected anomaly (legacy-identical)."""
    if kind == "temporal_activity_contradiction":
        casas = {
            "predicted_label": "Cook", "confidence": 0.92,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"sensor_tokens": ["M01:ON", "M02:ON", "AD1-A:ON"],
                         "model_tag": "gru_federation_test", "source_dataset": "casas"},
        }
        sphere = {
            "predicted_label": "p_lie", "confidence": 0.88,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"active_pirs": ["bed1"],
                         "model_tag": "lstm_federation_test", "source_dataset": "sphere"},
        }
    elif kind == "spatial_motion_inconsistency":
        casas = {
            "predicted_label": "Cook", "confidence": 0.85,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"sensor_tokens": ["M01:ON", "M02:ON", "AD1-B:ON"],
                         "model_tag": "gru_federation_test", "source_dataset": "casas"},
        }
        sphere = {
            "predicted_label": "a_walk", "confidence": 0.80,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"active_pirs": ["living", "hall"],
                         "model_tag": "lstm_federation_test", "source_dataset": "sphere"},
        }
    elif kind == "activity_posture_mismatch":
        casas = {
            "predicted_label": "WashHands", "confidence": 0.90,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"sensor_tokens": ["M08:ON", "D02:OPEN"],
                         "model_tag": "gru_federation_test", "source_dataset": "casas"},
        }
        sphere = {
            "predicted_label": "p_lie", "confidence": 0.87,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"active_pirs": ["bed1", "bed2"],
                         "model_tag": "lstm_federation_test", "source_dataset": "sphere"},
        }
    else:
        raise ValueError(f"unknown injection type: {kind}")
    return casas, sphere


def _type_allocation(n_anomalies: int) -> Dict[str, int]:
    """Per-type counts, matching the legacy allocation exactly."""
    n_each = max(1, n_anomalies // 3)
    return {
        "temporal_activity_contradiction": n_each,
        "spatial_motion_inconsistency": n_each,
        "activity_posture_mismatch": n_anomalies - 2 * n_each,
    }


def _inject_with_registry(
    casas_preds: List[Dict[str, Any]],
    sphere_preds: List[Dict[str, Any]],
    rng: random.Random,
    n_anomalies: int,
    legacy_times: bool,
) -> List[Dict[str, Any]]:
    """Append injected prediction pairs and return the ground-truth registry.

    With ``legacy_times=True`` the fixed 12:00/14:00/18:00 grid of the legacy
    script is reproduced (used by the regression check). Otherwise timestamps
    are drawn uniformly over the simulated day, collision-checked against all
    background timestamps and other injections so registry matching by
    timestamp/URI is unambiguous.
    """
    allocation = _type_allocation(n_anomalies)
    used_ts: Set[str] = {p["window_start"] for p in casas_preds}
    used_ts |= {p["window_start"] for p in sphere_preds}

    registry: List[Dict[str, Any]] = []
    for kind in _INJECTION_ORDER:
        for i in range(allocation[kind]):
            if legacy_times:
                ts = BASE_DATE.replace(hour=_LEGACY_BASE_HOURS[kind]) + timedelta(minutes=i * 5)
                ts_str = ts.isoformat()
            else:
                while True:
                    ts_str = (BASE_DATE + timedelta(seconds=rng.randint(0, 24 * 3600 - 1))).isoformat()
                    if ts_str not in used_ts:
                        break
            used_ts.add(ts_str)

            casas_payload, sphere_payload = _injection_payloads(kind, ts_str)
            casas_idx = len(casas_preds)
            sphere_idx = len(sphere_preds)
            casas_preds.append(casas_payload)
            sphere_preds.append(sphere_payload)
            registry.append({
                "id": f"inj_{len(registry):03d}",
                "type": kind,
                "timestamp": ts_str,
                "casas_index": casas_idx,
                "sphere_index": sphere_idx,
            })
    return registry


# ---------------------------------------------------------------------------
# Generative ground-truth labels
# ---------------------------------------------------------------------------

def _casas_motion_rooms(pred: Dict[str, Any], ds_map: Dict[str, Any]) -> Set[str]:
    """Rooms with motion-detected sensors in this prediction's window (mirrors loader semantics)."""
    patterns = ds_map.get("sensor_type_patterns", {})
    room_map = ds_map.get("room_assignments", {})
    rooms: Set[str] = set()
    seen: Set[str] = set()
    for token in pred.get("metadata", {}).get("sensor_tokens", []):
        parts = str(token).split(":", 1) if ":" in str(token) else str(token).split("=", 1)
        if len(parts) != 2:
            continue
        sensor_id, state_val = parts[0].strip(), parts[1].strip()
        if sensor_id in seen:
            continue
        seen.add(sensor_id)
        type_info = _resolve_sensor_type(sensor_id, patterns)
        if not type_info:
            continue
        mapped = type_info.get("state_map", {}).get(state_val.upper())
        if mapped == "nsiot:StateMotionDetected" and sensor_id in room_map:
            rooms.add(room_map[sensor_id])
    return rooms


def _sphere_motion_rooms(pred: Dict[str, Any], ds_map: Dict[str, Any]) -> Set[str]:
    pir_columns = ds_map.get("pir_columns", {})
    rooms: Set[str] = set()
    for pir in pred.get("metadata", {}).get("active_pirs", []):
        info = pir_columns.get(pir)
        if info and info.get("room"):
            rooms.add(info["room"])
    return rooms


def _generative_labels(
    preds: Sequence[Dict[str, Any]],
    source: str,
    ds_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Index-based ground-truth labels derived from the generator output itself."""
    labels: Dict[str, Any] = {
        "low_confidence": set(),
        "night": set(),
        "multi_room": set(),
        "motion_rooms": {},
    }
    for idx, pred in enumerate(preds):
        # Mirror the SPARQL engine's semantics: the KG stores confidence as a
        # binary float, the filter compares it against decimal 0.60, so a
        # confidence rounded to exactly 0.6 (binary 0.59999999999999997...)
        # falls below the threshold.
        if Decimal(float(pred.get("confidence", 0.0))) < Decimal("0.60"):
            labels["low_confidence"].add(idx)
        if _is_night_time(pred.get("window_start")):
            labels["night"].add(idx)
        rooms = (
            _casas_motion_rooms(pred, ds_map) if source == "casas"
            else _sphere_motion_rooms(pred, ds_map)
        )
        labels["motion_rooms"][idx] = rooms
        if len(rooms) >= 2:
            labels["multi_room"].add(idx)
    return labels


def _augment_graph_with_rooms(g: Graph, labels: Dict[str, Any], event_prefix: str) -> None:
    """Add event->room motion provenance triples enabling event-scoped queries."""
    involves = NSIOT["involvesMotionInRoom"]
    for idx, rooms in labels["motion_rooms"].items():
        event_uri = NSIOT[f"{event_prefix}{idx}"]
        for room in rooms:
            g.add((event_uri, involves, NSIOT[room.split(":", 1)[1]]))


# ---------------------------------------------------------------------------
# Queries. The first six are copied verbatim from the legacy detect functions
# (same WHERE clauses => provably identical counts); the last two are the
# corrected event-scoped variants.
# ---------------------------------------------------------------------------

Q_LOW_CONFIDENCE = (
    "SELECT ?pred ?conf WHERE { ?pred <http://example.org/neuro-symbolic-iot#hasConfidenceScore> ?conf . "
    "FILTER(?conf < 0.60) }"
)

Q_NIGHT_EVENTS = """SELECT ?e WHERE {
    ?e a <http://example.org/neuro-symbolic-iot#Event> .
    ?e <http://example.org/neuro-symbolic-iot#hasTimeContext> ?ctx .
    ?ctx a <http://example.org/neuro-symbolic-iot#NightTimeContext> .
}"""

Q_MULTIROOM_STATIC = """SELECT ?room1 ?room2 WHERE {
    ?s1 <http://example.org/neuro-symbolic-iot#isLocatedIn> ?room1 .
    ?s1 <http://example.org/neuro-symbolic-iot#hasState> <http://example.org/neuro-symbolic-iot#StateMotionDetected> .
    ?s2 <http://example.org/neuro-symbolic-iot#isLocatedIn> ?room2 .
    ?s2 <http://example.org/neuro-symbolic-iot#hasState> <http://example.org/neuro-symbolic-iot#StateMotionDetected> .
    FILTER(?room1 != ?room2 && STR(?room1) < STR(?room2))
}"""

Q_TEMPORAL_OVERLAP = """SELECT ?e1 ?e2 WHERE {
    ?e1 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_CASAS> .
    ?e2 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_SPHERE> .
    ?e1 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te1 .
    ?e2 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te2 .
    ?te1 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t1 .
    ?te2 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t2 .
    FILTER(?t1 = ?t2)
}"""

Q_ACTIVITY_POSTURE = """SELECT ?cp ?sp WHERE {
    ?cp a <http://example.org/neuro-symbolic-iot#NeuralPrediction> .
    ?cp <http://example.org/neuro-symbolic-iot#generatedByModel> ?m1 .
    FILTER(CONTAINS(STR(?m1), "gru"))
    ?cp <http://example.org/neuro-symbolic-iot#predictsActivity> ?act .
    ?cp <http://example.org/neuro-symbolic-iot#generatedAt> ?t .
    ?cp <http://example.org/neuro-symbolic-iot#hasConfidenceScore> ?c1 .
    FILTER(?c1 > 0.80)
    ?sp a <http://example.org/neuro-symbolic-iot#NeuralPrediction> .
    ?sp <http://example.org/neuro-symbolic-iot#generatedByModel> ?m2 .
    FILTER(CONTAINS(STR(?m2), "lstm"))
    ?sp <http://example.org/neuro-symbolic-iot#predictsActivity> ?post .
    ?sp <http://example.org/neuro-symbolic-iot#generatedAt> ?t .
    ?sp <http://example.org/neuro-symbolic-iot#hasConfidenceScore> ?c2 .
    FILTER(?c2 > 0.80)
    FILTER(
        (STR(?act) = "http://example.org/neuro-symbolic-iot#MealPreparation" && STR(?post) = "http://example.org/neuro-symbolic-iot#Lying") ||
        (STR(?act) = "http://example.org/neuro-symbolic-iot#PersonalHygiene" && STR(?post) = "http://example.org/neuro-symbolic-iot#Lying") ||
        (STR(?act) = "http://example.org/neuro-symbolic-iot#Housekeeping" && STR(?post) = "http://example.org/neuro-symbolic-iot#Lying")
    )
}"""

Q_LOCATION_STATIC = """SELECT ?s1 ?r1 ?s2 ?r2 WHERE {
    ?s1 <http://example.org/neuro-symbolic-iot#isLocatedIn> ?r1 .
    ?s1 <http://example.org/neuro-symbolic-iot#hasState> <http://example.org/neuro-symbolic-iot#StateMotionDetected> .
    ?s2 <http://example.org/neuro-symbolic-iot#isLocatedIn> ?r2 .
    ?s2 <http://example.org/neuro-symbolic-iot#hasState> <http://example.org/neuro-symbolic-iot#StateMotionDetected> .
    FILTER(STR(?s1) < STR(?s2))
    FILTER(?r1 != ?r2)
    FILTER(
        (CONTAINS(STR(?s1), "sensor_") && CONTAINS(STR(?s2), "pir_")) ||
        (CONTAINS(STR(?s1), "pir_") && CONTAINS(STR(?s2), "sensor_"))
    )
}"""

Q_MULTIROOM_EVENT = """SELECT DISTINCT ?e WHERE {
    ?e <http://example.org/neuro-symbolic-iot#involvesMotionInRoom> ?r1 .
    ?e <http://example.org/neuro-symbolic-iot#involvesMotionInRoom> ?r2 .
    FILTER(?r1 != ?r2)
}"""

Q_LOCATION_EVENT = """SELECT DISTINCT ?e1 ?e2 WHERE {
    ?e1 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_CASAS> .
    ?e2 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_SPHERE> .
    ?e1 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te1 .
    ?e2 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te2 .
    ?te1 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t1 .
    ?te2 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t2 .
    FILTER(?t1 = ?t2)
    ?e1 <http://example.org/neuro-symbolic-iot#involvesMotionInRoom> ?r1 .
    ?e2 <http://example.org/neuro-symbolic-iot#involvesMotionInRoom> ?r2 .
    FILTER NOT EXISTS {
        ?e1 <http://example.org/neuro-symbolic-iot#involvesMotionInRoom> ?shared .
        ?e2 <http://example.org/neuro-symbolic-iot#involvesMotionInRoom> ?shared .
    }
}"""

_POSTURE_CONFLICT_PAIRS = {
    ("nsiot:MealPreparation", "nsiot:Lying"),
    ("nsiot:PersonalHygiene", "nsiot:Lying"),
    ("nsiot:Housekeeping", "nsiot:Lying"),
}


def _rows(g: Graph, sparql: str) -> List[Tuple[str, ...]]:
    return [tuple(str(v) for v in row) for row in g.query(sparql)]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _resolve_activity(raw: str, activity_map: Dict[str, str]) -> str:
    label = activity_map.get(raw, raw)
    return label if label.startswith("nsiot:") else f"nsiot:{label}"


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1: Optional[float] = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _score_pairs(
    detected: Set[Tuple[str, str]],
    truth_by_pair: Dict[Tuple[str, str], str],
) -> Tuple[Dict[str, Any], Set[str], Set[Tuple[str, str]]]:
    """Score detected URI pairs against expected pairs; return PRF, matched ids, FP pairs."""
    truth_pairs = set(truth_by_pair)
    tp_pairs = detected & truth_pairs
    fp_pairs = detected - truth_pairs
    fn_pairs = truth_pairs - detected
    matched_ids = {truth_by_pair[p] for p in tp_pairs}
    return _prf(len(tp_pairs), len(fp_pairs), len(fn_pairs)), matched_ids, fp_pairs


def _score_entities(detected: Set[str], truth: Set[str]) -> Dict[str, Any]:
    return _prf(len(detected & truth), len(detected - truth), len(truth - detected))


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

def run_trial(
    sensor_map: Dict[str, Any],
    n_casas: int,
    n_sphere: int,
    n_anomalies: int,
    trial: int,
    seed: int,
    legacy_times: bool = False,
) -> Dict[str, Any]:
    """One randomized trial: generate, inject, build graphs, detect, score."""
    rng = random.Random(seed + trial * 1000)
    casas_preds = _generate_casas_predictions(n_casas, rng, sensor_map, BASE_DATE)
    sphere_preds = _generate_sphere_predictions(n_sphere, rng, sensor_map, BASE_DATE)
    registry = _inject_with_registry(casas_preds, sphere_preds, rng, n_anomalies, legacy_times)

    ds_casas = sensor_map.get("casas", {})
    ds_sphere = sensor_map.get("sphere", {})
    labels_casas = _generative_labels(casas_preds, "casas", ds_casas)
    labels_sphere = _generative_labels(sphere_preds, "sphere", ds_sphere)

    g_casas = build_kg_from_predictions(casas_preds, sensor_map, "casas")
    g_sphere = build_kg_from_predictions(sphere_preds, sensor_map, "sphere")
    g_fed = build_federated_kg(casas_preds, sphere_preds, sensor_map)

    _augment_graph_with_rooms(g_casas, labels_casas, "event_")
    _augment_graph_with_rooms(g_sphere, labels_sphere, "event_")
    _augment_graph_with_rooms(g_fed, labels_casas, "event_")
    _augment_graph_with_rooms(g_fed, labels_sphere, "sphere_event_")

    # --- run every query once, capturing bindings ---
    single_rows = {}
    for name, g in (("casas", g_casas), ("sphere", g_sphere)):
        single_rows[name] = {
            "low_confidence": _rows(g, Q_LOW_CONFIDENCE),
            "night_time_events": _rows(g, Q_NIGHT_EVENTS),
            "multi_room_motion": _rows(g, Q_MULTIROOM_STATIC),
            "multi_room_events": _rows(g, Q_MULTIROOM_EVENT),
        }
    fed_rows = {
        "low_confidence": _rows(g_fed, Q_LOW_CONFIDENCE),
        "night_time_events": _rows(g_fed, Q_NIGHT_EVENTS),
        "multi_room_motion": _rows(g_fed, Q_MULTIROOM_STATIC),
        "multi_room_events": _rows(g_fed, Q_MULTIROOM_EVENT),
        "temporal_overlap": _rows(g_fed, Q_TEMPORAL_OVERLAP),
        "activity_posture": _rows(g_fed, Q_ACTIVITY_POSTURE),
        "location_static": _rows(g_fed, Q_LOCATION_STATIC),
        "location_event": _rows(g_fed, Q_LOCATION_EVENT),
    }

    # --- as-published metric path (legacy semantics, incl. its double count) ---
    def _published_single(rows: Dict[str, List]) -> Dict[str, int]:
        out = {
            "low_confidence": len(rows["low_confidence"]),
            "night_time_events": len(rows["night_time_events"]),
            "multi_room_motion": len(rows["multi_room_motion"]),
        }
        out["total"] = sum(out.values())
        return out

    pub_casas = _published_single(single_rows["casas"])
    pub_sphere = _published_single(single_rows["sphere"])
    pub_fed = _published_single(fed_rows)
    pub_fed["cross_source_temporal_overlap"] = len(fed_rows["temporal_overlap"])
    pub_fed["cross_source_activity_posture_conflict"] = len(fed_rows["activity_posture"])
    pub_fed["cross_source_location_conflict"] = len(fed_rows["location_static"])
    pub_fed["cross_source_total"] = (
        pub_fed["cross_source_temporal_overlap"]
        + pub_fed["cross_source_activity_posture_conflict"]
        + pub_fed["cross_source_location_conflict"]
    )
    pub_fed["total"] = pub_fed["total"] + pub_fed["cross_source_total"]

    pub_union = pub_casas["total"] + pub_sphere["total"]
    pub_shared = (
        pub_fed["low_confidence"] + pub_fed["night_time_events"] + pub_fed["multi_room_motion"]
    )
    as_published = {
        "casas": pub_casas,
        "sphere": pub_sphere,
        "federated": pub_fed,
        "union_total": pub_union,
        "overall_lift_pct": 100.0 * (pub_fed["total"] - pub_union) / pub_union if pub_union else None,
        "shared_category_lift_pct": 100.0 * (pub_shared - pub_union) / pub_union if pub_union else None,
    }

    # --- ground-truth sets in federated-graph URI space ---
    def _uri(prefix: str, idx: int) -> str:
        return f"{NSIOT_STR}{prefix}{idx}"

    activity_map_casas = ds_casas.get("activity_map", {})
    activity_map_sphere = ds_sphere.get("activity_map", {})

    truth_temporal: Dict[Tuple[str, str], str] = {}
    truth_posture: Dict[Tuple[str, str], str] = {}
    truth_location: Dict[Tuple[str, str], str] = {}
    for entry in registry:
        ci, si = entry["casas_index"], entry["sphere_index"]
        event_pair = (_uri("event_", ci), _uri("sphere_event_", si))
        pred_pair = (_uri("pred_", ci), _uri("sphere_pred_", si))
        truth_temporal[event_pair] = entry["id"]

        c_pred, s_pred = casas_preds[ci], sphere_preds[si]
        act = _resolve_activity(c_pred["predicted_label"], activity_map_casas)
        post = _resolve_activity(s_pred["predicted_label"], activity_map_sphere)
        if (
            c_pred["confidence"] > 0.80
            and s_pred["confidence"] > 0.80
            and (act, post) in _POSTURE_CONFLICT_PAIRS
        ):
            truth_posture[pred_pair] = entry["id"]

        rooms_c = labels_casas["motion_rooms"][ci]
        rooms_s = labels_sphere["motion_rooms"][si]
        if rooms_c and rooms_s and not (rooms_c & rooms_s):
            truth_location[event_pair] = entry["id"]

    prf_temporal, ids_temporal, fp_temporal = _score_pairs(
        {(r[0], r[1]) for r in fed_rows["temporal_overlap"]}, truth_temporal
    )
    prf_posture, ids_posture, fp_posture = _score_pairs(
        {(r[0], r[1]) for r in fed_rows["activity_posture"]}, truth_posture
    )
    prf_location, ids_location, fp_location = _score_pairs(
        {(r[0], r[1]) for r in fed_rows["location_event"]}, truth_location
    )

    # single-source definitional categories, scored on the federated graph
    truth_low = {_uri("pred_", i) for i in labels_casas["low_confidence"]}
    truth_low |= {_uri("sphere_pred_", i) for i in labels_sphere["low_confidence"]}
    truth_night = {_uri("event_", i) for i in labels_casas["night"]}
    truth_night |= {_uri("sphere_event_", i) for i in labels_sphere["night"]}
    truth_multiroom = {_uri("event_", i) for i in labels_casas["multi_room"]}
    truth_multiroom |= {_uri("sphere_event_", i) for i in labels_sphere["multi_room"]}

    prf_low = _score_entities({r[0] for r in fed_rows["low_confidence"]}, truth_low)
    prf_night = _score_entities({r[0] for r in fed_rows["night_time_events"]}, truth_night)
    prf_multiroom = _score_entities({r[0] for r in fed_rows["multi_room_events"]}, truth_multiroom)

    # --- corrected metric path (event-scoped, de-duplicated) ---
    def _corrected_single(rows: Dict[str, List]) -> Dict[str, int]:
        out = {
            "low_confidence": len(rows["low_confidence"]),
            "night_time_events": len(rows["night_time_events"]),
            "multi_room_events": len(rows["multi_room_events"]),
        }
        out["total"] = sum(out.values())
        return out

    corr_casas = _corrected_single(single_rows["casas"])
    corr_sphere = _corrected_single(single_rows["sphere"])
    corr_union = corr_casas["total"] + corr_sphere["total"]
    corr_fed_shared = {
        "low_confidence": len(fed_rows["low_confidence"]),
        "night_time_events": len(fed_rows["night_time_events"]),
        "multi_room_events": len(fed_rows["multi_room_events"]),
    }
    corr_cross = {
        "temporal_overlap": len(fed_rows["temporal_overlap"]),
        "activity_posture": len(fed_rows["activity_posture"]),
        "location_event_pairs": len(fed_rows["location_event"]),
    }
    corr_cross_rows = sum(corr_cross.values())
    corr_fed_total = sum(corr_fed_shared.values()) + corr_cross_rows

    detected_ids = ids_temporal | ids_posture | ids_location

    def _as_event_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
        return (
            pair[0].replace("pred_", "event_") if "pred_" in pair[0] else pair[0],
            pair[1].replace("sphere_pred_", "sphere_event_") if "sphere_pred_" in pair[1] else pair[1],
        )

    fp_pairs_distinct = (
        {p for p in fp_temporal}
        | {_as_event_pair(p) for p in fp_posture}
        | {p for p in fp_location}
    )
    distinct_new = len(detected_ids) + len(fp_pairs_distinct)

    corrected = {
        "casas": corr_casas,
        "sphere": corr_sphere,
        "union_total": corr_union,
        "federated_shared": corr_fed_shared,
        "federated_cross": corr_cross,
        "federated_cross_rows": corr_cross_rows,
        "federated_total": corr_fed_total,
        "overall_lift_rows_pct": 100.0 * (corr_fed_total - corr_union) / corr_union if corr_union else None,
        "shared_category_lift_pct": (
            100.0 * (sum(corr_fed_shared.values()) - corr_union) / corr_union if corr_union else None
        ),
        "distinct_new_anomalies": distinct_new,
        "distinct_lift_pct": 100.0 * distinct_new / corr_union if corr_union else None,
        "registry_detected": len(detected_ids),
        "registry_size": len(registry),
    }

    return {
        "trial": trial,
        "trial_seed": seed + trial * 1000,
        "n_injected": len(registry),
        "as_published": as_published,
        "corrected": corrected,
        "prf": {
            "cross_source_temporal_overlap": prf_temporal,
            "cross_source_activity_posture_conflict": prf_posture,
            "cross_source_location_conflict_event_scoped": prf_location,
            "low_confidence": prf_low,
            "night_time_events": prf_night,
            "multi_room_motion_event_scoped": prf_multiroom,
        },
        "triples": {"casas": len(g_casas), "sphere": len(g_sphere), "federated": len(g_fed)},
        "registry": registry,
    }


# ---------------------------------------------------------------------------
# Regression check against the legacy trial 0
# ---------------------------------------------------------------------------

LEGACY_TRIAL0_EXPECTED = {
    "casas_total": 74,
    "sphere_total": 95,
    "federated_total": 308,
    "cross_source_temporal_overlap": 10,
    "cross_source_activity_posture_conflict": 7,
    "cross_source_location_conflict": 61,
}


def run_regression_check(sensor_map: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Reproduce legacy trial 0 (fixed injection grid) and verify both metric paths.

    1. The binding-based ``as_published`` path must equal the legacy
       ``_detect_anomalies_*`` outputs on the same graphs.
    2. Both must equal the stored trial-0 numbers behind the manuscript.
    """
    rng = random.Random(seed)
    casas_preds = _generate_casas_predictions(100, rng, sensor_map, BASE_DATE)
    sphere_preds = _generate_sphere_predictions(100, rng, sensor_map, BASE_DATE)
    _inject_with_registry(casas_preds, sphere_preds, rng, 10, legacy_times=True)

    g_casas = build_kg_from_predictions(casas_preds, sensor_map, "casas")
    g_sphere = build_kg_from_predictions(sphere_preds, sensor_map, "sphere")
    g_fed = build_federated_kg(casas_preds, sphere_preds, sensor_map)

    legacy = {
        "casas": _detect_anomalies_single(g_casas, "casas"),
        "sphere": _detect_anomalies_single(g_sphere, "sphere"),
        "federated": _detect_anomalies_federated(g_fed),
    }

    trial = run_trial(sensor_map, 100, 100, 10, trial=0, seed=seed, legacy_times=True)
    mine = trial["as_published"]

    mismatches: List[str] = []
    for src in ("casas", "sphere"):
        if mine[src] != legacy[src]:
            mismatches.append(f"{src}: mine={mine[src]} legacy={legacy[src]}")
    if mine["federated"] != legacy["federated"]:
        mismatches.append(f"federated: mine={mine['federated']} legacy={legacy['federated']}")

    observed = {
        "casas_total": legacy["casas"]["total"],
        "sphere_total": legacy["sphere"]["total"],
        "federated_total": legacy["federated"]["total"],
        "cross_source_temporal_overlap": legacy["federated"]["cross_source_temporal_overlap"],
        "cross_source_activity_posture_conflict": legacy["federated"]["cross_source_activity_posture_conflict"],
        "cross_source_location_conflict": legacy["federated"]["cross_source_location_conflict"],
    }
    for key, expected in LEGACY_TRIAL0_EXPECTED.items():
        if observed[key] != expected:
            mismatches.append(f"{key}: observed={observed[key]} expected={expected}")

    if mismatches:
        raise RuntimeError("Regression check FAILED:\n  " + "\n  ".join(mismatches))

    log.info("Regression check passed: as_published path == legacy path == stored trial-0 numbers")
    return {"passed": True, "observed": observed, "trial": trial, "legacy": legacy}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(values: Sequence[Optional[float]], ndigits: int = 2) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    arr = np.asarray(vals)
    mean = float(arr.mean())
    if n > 1:
        std = float(arr.std(ddof=1))
        half = float(scistats.t.ppf(0.975, n - 1)) * std / float(np.sqrt(n))
    else:
        std, half = 0.0, 0.0
    return {
        "mean": round(mean, ndigits),
        "std": round(std, ndigits),
        "ci95": [round(mean - half, ndigits), round(mean + half, ndigits)],
        "n": n,
        "per_trial": [round(v, ndigits) for v in vals],
    }


def _agg_prf(trials: Sequence[Dict[str, Any]], category: str) -> Dict[str, Any]:
    tp = sum(t["prf"][category]["tp"] for t in trials)
    fp = sum(t["prf"][category]["fp"] for t in trials)
    fn = sum(t["prf"][category]["fn"] for t in trials)
    pooled = _prf(tp, fp, fn)
    out = {
        "tp": _agg([t["prf"][category]["tp"] for t in trials]),
        "fp": _agg([t["prf"][category]["fp"] for t in trials]),
        "fn": _agg([t["prf"][category]["fn"] for t in trials]),
        "precision": _agg([t["prf"][category]["precision"] for t in trials], 4),
        "recall": _agg([t["prf"][category]["recall"] for t in trials], 4),
        "f1": _agg([t["prf"][category]["f1"] for t in trials], 4),
        "pooled": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(pooled["precision"], 4) if pooled["precision"] is not None else None,
            "recall": round(pooled["recall"], 4) if pooled["recall"] is not None else None,
            "f1": round(pooled["f1"], 4) if pooled["f1"] is not None else None,
        },
    }
    return out


def _aggregate_arm(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "as_published": {
            "casas_total": _agg([t["as_published"]["casas"]["total"] for t in trials]),
            "sphere_total": _agg([t["as_published"]["sphere"]["total"] for t in trials]),
            "union_total": _agg([t["as_published"]["union_total"] for t in trials]),
            "federated_total": _agg([t["as_published"]["federated"]["total"] for t in trials]),
            "cross_source_total": _agg(
                [t["as_published"]["federated"]["cross_source_total"] for t in trials]
            ),
            "overall_lift_pct": _agg([t["as_published"]["overall_lift_pct"] for t in trials]),
            "shared_category_lift_pct": _agg(
                [t["as_published"]["shared_category_lift_pct"] for t in trials]
            ),
        },
        "corrected": {
            "casas_total": _agg([t["corrected"]["casas"]["total"] for t in trials]),
            "sphere_total": _agg([t["corrected"]["sphere"]["total"] for t in trials]),
            "union_total": _agg([t["corrected"]["union_total"] for t in trials]),
            "federated_total": _agg([t["corrected"]["federated_total"] for t in trials]),
            "federated_cross_rows": _agg([t["corrected"]["federated_cross_rows"] for t in trials]),
            "overall_lift_rows_pct": _agg([t["corrected"]["overall_lift_rows_pct"] for t in trials]),
            "shared_category_lift_pct": _agg(
                [t["corrected"]["shared_category_lift_pct"] for t in trials]
            ),
            "distinct_new_anomalies": _agg([t["corrected"]["distinct_new_anomalies"] for t in trials]),
            "distinct_lift_pct": _agg([t["corrected"]["distinct_lift_pct"] for t in trials]),
            "registry_detected": _agg([t["corrected"]["registry_detected"] for t in trials]),
        },
        "prf": {cat: _agg_prf(trials, cat) for cat in trials[0]["prf"]},
    }


# ---------------------------------------------------------------------------
# Metadata / definitions
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


DEFINITIONS = {
    "as_published": (
        "Legacy metric path, byte-identical query semantics to "
        "evaluation/run_cross_dataset_federation.py. Known artefacts preserved for "
        "reconciliation: (1) 'multi_room_motion' and 'cross_source_location_conflict' are "
        "graph-static sensor-pair counts, not event detections; (2) in the federated graph "
        "the cross-source sensor pairs are counted under BOTH categories (double count); "
        "(3) single-source categories are definitional pattern matches on the synthetic "
        "background, not independently validated anomalies."
    ),
    "corrected_rows": (
        "Event-scoped, de-duplicated detections. 'multi_room_events' counts events whose "
        "motion-active sensors span >=2 rooms (via nsiot:involvesMotionInRoom provenance "
        "triples added by this script); 'location_event_pairs' counts CASAS/SPHERE event "
        "pairs with equal timestamps and disjoint motion-room sets. No detection is counted "
        "in more than one category slot of the total."
    ),
    "corrected_distinct": (
        "Distinct anomalies: number of ground-truth registry entries detected by at least "
        "one cross-source query, plus distinct unmatched (false-positive) event pairs. "
        "This is the strictest honest count of NEW anomalies federation contributes."
    ),
    "ground_truth": (
        "The experiment is fully synthetic, so ground truth is programmatic: injected "
        "anomalies carry ids/timestamps/indices in a registry (matched to detections by "
        "URI pair); background labels (confidence < 0.60, night-time 22:00-06:00, motion "
        "spanning >=2 rooms) are derived from the generator output itself. Single-source "
        "categories are therefore rule-definitional: precision 1.0 by construction, and "
        "they characterise the synthetic stream, not detector skill."
    ),
    "truth_per_category": {
        "cross_source_temporal_overlap": "all injected pairs (each shares an exact timestamp)",
        "cross_source_activity_posture_conflict": (
            "injected pairs with both confidences > 0.80 and (activity, posture) in "
            "{(MealPreparation, Lying), (PersonalHygiene, Lying), (Housekeeping, Lying)}"
        ),
        "cross_source_location_conflict_event_scoped": (
            "injected pairs whose CASAS and SPHERE motion-room sets are non-empty and disjoint"
        ),
        "low_confidence": (
            "predictions whose stored binary-float confidence is below the decimal "
            "threshold 0.60 — this includes values rounded to exactly 0.6, whose float "
            "representation (0.59999999999999997...) the SPARQL engine correctly ranks "
            "below the decimal literal 0.60"
        ),
        "night_time_events": "events generated with timestamp in 22:00-06:00",
        "multi_room_motion_event_scoped": "events whose motion-active sensors span >= 2 rooms",
    },
    "single_source_recall_on_cross_source_categories": (
        "0.0 by construction: cross-source anomaly pairs span two graphs that a "
        "single-source deployment never joins, so no single-source query can detect them."
    ),
    "seed_scheme": "trial seed = seed + 1000 * trial (same scheme as the CV folds)",
    "triple_count_note": (
        "Federated KG is now ~3110 triples at trial 0 vs 2890 in the stored legacy run: "
        "+1 triple/prediction from the isAlternativePrediction flag later added to "
        "kg_federation_loader.py, plus this script's involvesMotionInRoom provenance "
        "triples on its own graphs. Detection counts are unaffected."
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1: multi-trial, ground-truth-scored federation experiment.",
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--n-casas", type=int, default=100, help="CASAS background predictions")
    parser.add_argument("--n-sphere", type=int, default=100, help="SPHERE background predictions")
    parser.add_argument("--trials", type=int, default=20, help="Randomized trials per injection arm")
    parser.add_argument(
        "--anomaly-counts", default="5,10,20,40",
        help="Comma-separated injected-anomaly counts (sensitivity arms)",
    )
    parser.add_argument(
        "--headline-count", type=int, default=10,
        help="Injection arm used for multi_trial.json and per_category_prf.json",
    )
    parser.add_argument("--outdir", default="evaluation/results_federation", help="Output dir")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--skip-regression", action="store_true", help="Skip legacy trial-0 check")
    return parser.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging
    from neurosymbolic_iot.utils.seed import set_global_seed

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_global_seed(args.seed)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kg_cfg = cfg.get("kg", {})
    sensor_map_path = Path(
        kg_cfg.get("sensor_map_path", "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json")
    )
    sensor_map = load_sensor_map(sensor_map_path)

    arms = [int(x) for x in args.anomaly_counts.split(",") if x.strip()]
    if args.headline_count not in arms:
        arms.append(args.headline_count)
    arms = sorted(set(arms))

    reconciliation: Optional[Dict[str, Any]] = None
    if not args.skip_regression:
        log.info("Running regression check against legacy trial 0 ...")
        reg = run_regression_check(sensor_map, args.seed)
        legacy_fed = reg["legacy"]["federated"]
        loc_static = legacy_fed["cross_source_location_conflict"]
        mr_fed = legacy_fed["multi_room_motion"]
        union0 = reg["legacy"]["casas"]["total"] + reg["legacy"]["sphere"]["total"]
        reconciliation = {
            "n_injected": 10,
            "as_published_trial0": {
                "federated_total": legacy_fed["total"],
                "union_of_single_source": union0,
                "casas_total": reg["legacy"]["casas"]["total"],
                "sphere_total": reg["legacy"]["sphere"]["total"],
            },
            "decomposition_of_federated_total": {
                "single_source_definitional_matches": {
                    "low_confidence": legacy_fed["low_confidence"],
                    "night_time_events": legacy_fed["night_time_events"],
                    "multi_room_motion_sensor_pairs": mr_fed,
                },
                "multi_room_motion_breakdown": {
                    "same_source_sensor_pairs": mr_fed - loc_static,
                    "cross_source_sensor_pairs_also_counted_as_location_conflict": loc_static,
                },
                "cross_source_rows": {
                    "temporal_overlap_pairs": legacy_fed["cross_source_temporal_overlap"],
                    "activity_posture_conflicts": legacy_fed["cross_source_activity_posture_conflict"],
                    "location_conflict_static_sensor_pairs": loc_static,
                },
            },
            "explanation": (
                "The N injected anomalies are prediction PAIRS sharing an exact timestamp; each "
                "surfaces as one temporal-overlap row, and subsets additionally satisfy the "
                "activity-posture and location-conflict patterns. The remaining detections are "
                "definitional pattern matches on the randomly generated background stream "
                "(confidence < 0.60, night-time timestamps, multi-room motion) plus graph-static "
                "sensor-pair counts. The as-published federated total also counts the cross-source "
                "location-conflict sensor pairs twice (once under multi-room motion, once under "
                "location conflict), which inflates the headline lift."
            ),
        }

    common_meta = {
        "script": "evaluation/run_federation_rigor.py",
        "commit": _git_commit(),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "n_casas": args.n_casas,
        "n_sphere": args.n_sphere,
        "trials_per_arm": args.trials,
        "anomaly_arms": arms,
        "headline_arm": args.headline_count,
        "seed": args.seed,
        "definitions": DEFINITIONS,
    }

    results_by_arm: Dict[int, List[Dict[str, Any]]] = {}
    for arm in arms:
        results_by_arm[arm] = []
        for trial in range(args.trials):
            t0 = time.perf_counter()
            record = run_trial(
                sensor_map, args.n_casas, args.n_sphere, arm, trial, args.seed,
            )
            results_by_arm[arm].append(record)
            log.info(
                "arm=%d trial=%d/%d done in %.1fs | as_published fed=%d lift=%.1f%% | "
                "corrected rows=%d distinct_new=%d registry %d/%d",
                arm, trial + 1, args.trials, time.perf_counter() - t0,
                record["as_published"]["federated"]["total"],
                record["as_published"]["overall_lift_pct"],
                record["corrected"]["federated_total"],
                record["corrected"]["distinct_new_anomalies"],
                record["corrected"]["registry_detected"],
                record["corrected"]["registry_size"],
            )

    agg_by_arm = {arm: _aggregate_arm(results_by_arm[arm]) for arm in arms}
    headline_trials = results_by_arm[args.headline_count]
    headline_agg = agg_by_arm[args.headline_count]

    # --- multi_trial.json ---
    multi_trial = {
        "_meta": {**common_meta, "n_anomalies": args.headline_count},
        "as_published": headline_agg["as_published"],
        "corrected": headline_agg["corrected"],
        "reconciliation": reconciliation,
        "raw_trials": [
            {k: v for k, v in t.items() if k != "registry"} for t in headline_trials
        ],
    }
    (out_dir / "multi_trial.json").write_text(
        json.dumps(multi_trial, indent=2, default=str), encoding="utf-8"
    )

    # --- per_category_prf.json ---
    per_category = {
        "_meta": {**common_meta, "n_anomalies": args.headline_count},
        "categories": headline_agg["prf"],
        "single_source_recall_on_cross_source_categories": 0.0,
        "note": DEFINITIONS["single_source_recall_on_cross_source_categories"],
    }
    (out_dir / "per_category_prf.json").write_text(
        json.dumps(per_category, indent=2, default=str), encoding="utf-8"
    )

    # --- injection_sensitivity.json ---
    sensitivity = {
        "_meta": common_meta,
        "arms": {
            str(arm): {
                "n_injected": arm,
                "as_published_overall_lift_pct": agg_by_arm[arm]["as_published"]["overall_lift_pct"],
                "corrected_overall_lift_rows_pct": agg_by_arm[arm]["corrected"]["overall_lift_rows_pct"],
                "corrected_distinct_lift_pct": agg_by_arm[arm]["corrected"]["distinct_lift_pct"],
                "corrected_distinct_new_anomalies": agg_by_arm[arm]["corrected"]["distinct_new_anomalies"],
                "registry_detected": agg_by_arm[arm]["corrected"]["registry_detected"],
                "cross_source_prf_pooled": {
                    cat: agg_by_arm[arm]["prf"][cat]["pooled"]
                    for cat in (
                        "cross_source_temporal_overlap",
                        "cross_source_activity_posture_conflict",
                        "cross_source_location_conflict_event_scoped",
                    )
                },
            }
            for arm in arms
        },
    }
    (out_dir / "injection_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2, default=str), encoding="utf-8"
    )

    # --- console summary ---
    ap = headline_agg["as_published"]
    co = headline_agg["corrected"]
    print()
    print("=" * 90)
    print("  E1 FEDERATION RIGOR — SUMMARY (headline arm: "
          f"{args.headline_count} injected, {args.trials} trials)")
    print("=" * 90)
    print(f"  as_published  federated total: {ap['federated_total']['mean']} ± {ap['federated_total']['std']}"
          f"  (95% CI {ap['federated_total']['ci95']})")
    print(f"  as_published  overall lift:    {ap['overall_lift_pct']['mean']}% ± {ap['overall_lift_pct']['std']}"
          f"  (95% CI {ap['overall_lift_pct']['ci95']})")
    print(f"  as_published  shared-cat lift: {ap['shared_category_lift_pct']['mean']}%")
    print(f"  corrected     federated rows:  {co['federated_total']['mean']} ± {co['federated_total']['std']}")
    print(f"  corrected     rows lift:       {co['overall_lift_rows_pct']['mean']}% ± {co['overall_lift_rows_pct']['std']}"
          f"  (95% CI {co['overall_lift_rows_pct']['ci95']})")
    print(f"  corrected     shared-cat lift: {co['shared_category_lift_pct']['mean']}%")
    print(f"  corrected     distinct new:    {co['distinct_new_anomalies']['mean']} ± {co['distinct_new_anomalies']['std']}"
          f"  -> distinct lift {co['distinct_lift_pct']['mean']}%")
    print(f"  registry detected:             {co['registry_detected']['mean']} / {args.headline_count}")
    print("  cross-source PRF (pooled):")
    for cat in ("cross_source_temporal_overlap", "cross_source_activity_posture_conflict",
                "cross_source_location_conflict_event_scoped"):
        pooled = headline_agg["prf"][cat]["pooled"]
        print(f"    {cat:<48} P={pooled['precision']} R={pooled['recall']} F1={pooled['f1']}")
    print("=" * 90)

    log.info("Wrote %s, %s, %s in %s",
             "multi_trial.json", "per_category_prf.json", "injection_sensitivity.json", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
