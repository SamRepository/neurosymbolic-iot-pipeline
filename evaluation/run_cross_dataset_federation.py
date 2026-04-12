"""
Cross-Dataset Federated Reasoning Experiment (Enhancement F)
============================================================
Populates a single KG with BOTH CASAS and SPHERE predictions,
runs federated SPARQL queries and anomaly-detection rules across
the merged graph, and compares against single-source baselines.

Usage:
  PYTHONPATH=. python evaluation/run_cross_dataset_federation.py --config config/base.yaml --trials 3
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
    build_kg_from_predictions,
    load_sensor_map,
)
from evaluation.metrics_collector import LatencyTracker

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
TIME = Namespace("http://www.w3.org/2006/time#")
NSIOT_STR = "http://example.org/neuro-symbolic-iot#"
TIME_STR = "http://www.w3.org/2006/time#"

_CASAS_SENSORS = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08",
    "D01", "D02", "D03", "AD1-A", "AD1-B", "AD1-C",
]
_CASAS_STATES = {"M": ["ON", "OFF"], "D": ["OPEN", "CLOSE"], "A": ["ON", "OFF"]}
_SPHERE_PIRS = ["bath", "bed1", "bed2", "hall", "kitchen", "living", "stairs", "study", "toilet"]


def _random_timestamp(rng, base_date):
    offset = timedelta(seconds=rng.randint(0, 24 * 3600))
    return (base_date + offset).isoformat()


def _generate_casas_predictions(n, rng, sensor_map, base_date):
    activities = list(sensor_map.get("casas", {}).get("activity_map", {}).keys())
    if not activities:
        activities = ["Cook", "Eat", "Clean", "PhoneCall", "WashHands"]
    preds = []
    for i in range(n):
        n_sensors = rng.randint(3, 6)
        sensors = rng.sample(_CASAS_SENSORS, min(n_sensors, len(_CASAS_SENSORS)))
        tokens = []
        for s in sensors:
            prefix = s[0]
            states = _CASAS_STATES.get(prefix, ["ON", "OFF"])
            tokens.append(f"{s}:{rng.choice(states)}")
        ts = _random_timestamp(rng, base_date)
        preds.append({
            "predicted_label": rng.choice(activities),
            "confidence": round(rng.uniform(0.50, 0.99), 3),
            "window_start": ts,
            "window_end": ts,
            "metadata": {
                "sensor_tokens": tokens,
                "model_tag": "gru_federation_test",
                "source_dataset": "casas",
            },
        })
    return preds


def _generate_sphere_predictions(n, rng, sensor_map, base_date):
    activities = list(sensor_map.get("sphere", {}).get("activity_map", {}).keys())
    if not activities:
        activities = ["p_stand", "p_sit", "p_lie", "a_walk"]
    preds = []
    for i in range(n):
        n_pirs = rng.randint(2, 4)
        pirs = rng.sample(_SPHERE_PIRS, min(n_pirs, len(_SPHERE_PIRS)))
        ts = _random_timestamp(rng, base_date)
        preds.append({
            "predicted_label": rng.choice(activities),
            "confidence": round(rng.uniform(0.50, 0.99), 3),
            "window_start": ts,
            "window_end": ts,
            "metadata": {
                "active_pirs": pirs,
                "model_tag": "lstm_federation_test",
                "source_dataset": "sphere",
            },
        })
    return preds


def _inject_cross_source_anomalies(casas_preds, sphere_preds, rng, n_anomalies=10):
    """Inject anomalies only detectable via cross-source reasoning."""
    anomalies = []
    n_each = max(1, n_anomalies // 3)

    # Type 1: Temporal activity contradiction (cooking + lying at same time)
    for i in range(n_each):
        shared_ts = datetime(2024, 6, 15, 12, 0, 0) + timedelta(minutes=i * 5)
        ts_str = shared_ts.isoformat()
        casas_preds.append({
            "predicted_label": "Cook", "confidence": 0.92,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"sensor_tokens": ["M01:ON", "M02:ON", "AD1-A:ON"],
                         "model_tag": "gru_federation_test", "source_dataset": "casas"},
        })
        sphere_preds.append({
            "predicted_label": "p_lie", "confidence": 0.88,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"active_pirs": ["bed1"],
                         "model_tag": "lstm_federation_test", "source_dataset": "sphere"},
        })
        anomalies.append({"type": "temporal_activity_contradiction", "timestamp": ts_str,
                         "casas_activity": "Cook", "sphere_posture": "p_lie"})

    # Type 2: Spatial motion inconsistency (kitchen vs living/hall)
    for i in range(n_each):
        shared_ts = datetime(2024, 6, 15, 14, 0, 0) + timedelta(minutes=i * 5)
        ts_str = shared_ts.isoformat()
        casas_preds.append({
            "predicted_label": "Cook", "confidence": 0.85,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"sensor_tokens": ["M01:ON", "M02:ON", "AD1-B:ON"],
                         "model_tag": "gru_federation_test", "source_dataset": "casas"},
        })
        sphere_preds.append({
            "predicted_label": "a_walk", "confidence": 0.80,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"active_pirs": ["living", "hall"],
                         "model_tag": "lstm_federation_test", "source_dataset": "sphere"},
        })
        anomalies.append({"type": "spatial_motion_inconsistency", "timestamp": ts_str,
                         "casas_location": "kitchen", "sphere_location": "living/hall"})

    # Type 3: Activity-posture mismatch (hygiene + lying)
    for i in range(n_anomalies - 2 * n_each):
        shared_ts = datetime(2024, 6, 15, 18, 0, 0) + timedelta(minutes=i * 5)
        ts_str = shared_ts.isoformat()
        casas_preds.append({
            "predicted_label": "WashHands", "confidence": 0.90,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"sensor_tokens": ["M08:ON", "D02:OPEN"],
                         "model_tag": "gru_federation_test", "source_dataset": "casas"},
        })
        sphere_preds.append({
            "predicted_label": "p_lie", "confidence": 0.87,
            "window_start": ts_str, "window_end": ts_str,
            "metadata": {"active_pirs": ["bed1", "bed2"],
                         "model_tag": "lstm_federation_test", "source_dataset": "sphere"},
        })
        anomalies.append({"type": "activity_posture_mismatch", "timestamp": ts_str,
                         "casas_activity": "WashHands", "sphere_posture": "p_lie"})

    return casas_preds, sphere_preds, anomalies


def build_federated_kg(casas_preds, sphere_preds, sensor_map):
    """Build a unified federated KG from both CASAS and SPHERE predictions."""
    g = Graph()
    g.bind("nsiot", NSIOT)
    g.bind("sosa", SOSA)
    g.bind("time", TIME)
    g.bind("owl", OWL)

    person_ref = NSIOT["Participant1"]
    g.add((person_ref, RDF.type, NSIOT["Person"]))

    casas_source = NSIOT["DataSource_CASAS"]
    sphere_source = NSIOT["DataSource_SPHERE"]
    g.add((casas_source, RDF.type, NSIOT["DataSource"]))
    g.add((casas_source, RDFS.label, Literal("CASAS Kyoto ADL")))
    g.add((sphere_source, RDF.type, NSIOT["DataSource"]))
    g.add((sphere_source, RDFS.label, Literal("SPHERE Challenge")))

    g_casas = build_kg_from_predictions(casas_preds, sensor_map, "casas")
    for t in g_casas:
        g.add(t)

    g_sphere = build_kg_from_predictions(sphere_preds, sensor_map, "sphere")
    offset = len(casas_preds)
    for s, p, o in g_sphere:
        s_new = _remap_uri(s, offset, "sphere")
        o_new = _remap_uri(o, offset, "sphere") if isinstance(o, URIRef) else o
        g.add((s_new, p, o_new))

    for i in range(len(casas_preds)):
        g.add((NSIOT[f"event_{i}"], NSIOT["hasDataSource"], casas_source))
    for i in range(len(sphere_preds)):
        g.add((NSIOT[f"sphere_event_{i}"], NSIOT["hasDataSource"], sphere_source))

    return g


def _remap_uri(uri, offset, prefix):
    """Remap URIs from a second dataset to avoid collision."""
    if not isinstance(uri, URIRef):
        return uri
    uri_str = str(uri)
    nsiot_str = str(NSIOT)
    if uri_str.startswith(nsiot_str):
        local = uri_str[len(nsiot_str):]
        for kind in ("pred_", "event_", "time_", "nightctx_"):
            if local.startswith(kind):
                try:
                    idx = int(local[len(kind):])
                    return NSIOT[f"{prefix}_{kind}{idx}"]
                except ValueError:
                    pass
    return uri


SINGLE_SOURCE_QUERIES = {
    "count_predictions": """
        SELECT (COUNT(?pred) AS ?n) WHERE {
            ?pred a <http://example.org/neuro-symbolic-iot#NeuralPrediction> .
        }
    """,
    "count_events": """
        SELECT (COUNT(?e) AS ?n) WHERE {
            ?e a <http://example.org/neuro-symbolic-iot#Event> .
        }
    """,
    "high_confidence_predictions": """
        SELECT ?pred ?conf WHERE {
            ?pred <http://example.org/neuro-symbolic-iot#hasConfidenceScore> ?conf .
            FILTER(?conf > 0.85)
        }
    """,
    "sensor_room_coverage": """
        SELECT DISTINCT ?room WHERE {
            ?s <http://example.org/neuro-symbolic-iot#isLocatedIn> ?room .
        }
    """,
    "events_with_temporal": """
        SELECT ?e ?time WHERE {
            ?e <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?t .
            ?t <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?time .
        }
    """,
}

FEDERATED_QUERIES = {
    "cross_source_temporal_overlap": """
        SELECT ?e1 ?e2 ?t1 ?t2 WHERE {
            ?e1 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_CASAS> .
            ?e2 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_SPHERE> .
            ?e1 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te1 .
            ?e2 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te2 .
            ?te1 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t1 .
            ?te2 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t2 .
            FILTER(?t1 = ?t2)
        }
    """,
    "cross_source_location_conflict": """
        SELECT ?s1 ?r1 ?s2 ?r2 WHERE {
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
        }
    """,
    "cross_source_activity_posture": """
        SELECT ?cp ?sp ?act ?post ?t WHERE {
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
        }
    """,
    "federated_density": """
        SELECT (COUNT(DISTINCT ?s) AS ?subjects)
               (COUNT(DISTINCT ?p) AS ?predicates)
               (COUNT(DISTINCT ?o) AS ?objects)
        WHERE { ?s ?p ?o . }
    """,
    "multi_source_high_confidence": """
        SELECT ?pred ?conf ?model WHERE {
            ?pred <http://example.org/neuro-symbolic-iot#hasConfidenceScore> ?conf .
            ?pred <http://example.org/neuro-symbolic-iot#generatedByModel> ?model .
            FILTER(?conf > 0.85)
        }
    """,
}


def _detect_anomalies_single(g, dataset):
    """Detect anomalies using single-source SPARQL queries."""
    results = {}

    q = "SELECT ?pred ?conf WHERE { ?pred <http://example.org/neuro-symbolic-iot#hasConfidenceScore> ?conf . FILTER(?conf < 0.60) }"
    results["low_confidence"] = len(list(g.query(q)))

    q = """SELECT ?e WHERE {
        ?e a <http://example.org/neuro-symbolic-iot#Event> .
        ?e <http://example.org/neuro-symbolic-iot#hasTimeContext> ?ctx .
        ?ctx a <http://example.org/neuro-symbolic-iot#NightTimeContext> .
    }"""
    results["night_time_events"] = len(list(g.query(q)))

    q = """SELECT ?room1 ?room2 WHERE {
        ?s1 <http://example.org/neuro-symbolic-iot#isLocatedIn> ?room1 .
        ?s1 <http://example.org/neuro-symbolic-iot#hasState> <http://example.org/neuro-symbolic-iot#StateMotionDetected> .
        ?s2 <http://example.org/neuro-symbolic-iot#isLocatedIn> ?room2 .
        ?s2 <http://example.org/neuro-symbolic-iot#hasState> <http://example.org/neuro-symbolic-iot#StateMotionDetected> .
        FILTER(?room1 != ?room2 && STR(?room1) < STR(?room2))
    }"""
    results["multi_room_motion"] = len(list(g.query(q)))

    results["total"] = results["low_confidence"] + results["night_time_events"] + results["multi_room_motion"]
    return results


def _detect_anomalies_federated(g):
    """Detect anomalies using federated cross-source SPARQL queries."""
    results = _detect_anomalies_single(g, "federated")

    q = """SELECT ?e1 ?e2 WHERE {
        ?e1 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_CASAS> .
        ?e2 <http://example.org/neuro-symbolic-iot#hasDataSource> <http://example.org/neuro-symbolic-iot#DataSource_SPHERE> .
        ?e1 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te1 .
        ?e2 <http://example.org/neuro-symbolic-iot#hasTemporalEntity> ?te2 .
        ?te1 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t1 .
        ?te2 <http://www.w3.org/2006/time#inXSDDateTimeStamp> ?t2 .
        FILTER(?t1 = ?t2)
    }"""
    results["cross_source_temporal_overlap"] = len(list(g.query(q)))

    q = """SELECT ?cp ?sp WHERE {
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
    results["cross_source_activity_posture_conflict"] = len(list(g.query(q)))

    q = """SELECT ?s1 ?r1 ?s2 ?r2 WHERE {
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
    results["cross_source_location_conflict"] = len(list(g.query(q)))

    results["cross_source_total"] = (
        results["cross_source_temporal_overlap"]
        + results["cross_source_activity_posture_conflict"]
        + results["cross_source_location_conflict"]
    )
    results["total"] = results["total"] + results["cross_source_total"]
    return results


def _run_queries(graph, queries, n_runs=10):
    """Run a query suite and return latency stats + result counts."""
    results = {}
    for qname, sparql in queries.items():
        tracker = LatencyTracker(name=qname)
        result_count = 0
        for run in range(n_runs):
            t0 = time.perf_counter()
            rows = list(graph.query(sparql))
            elapsed = time.perf_counter() - t0
            tracker.record(elapsed)
            if run == 0:
                result_count = len(rows)
        summary = tracker.summary()
        summary["result_count"] = result_count
        results[qname] = summary
    return results


def run_federation_trial(sensor_map, n_casas, n_sphere, n_anomalies, trial, seed=42):
    """Run one full trial: single-source vs federated."""
    rng = random.Random(seed + trial * 1000)
    base_date = datetime(2024, 6, 15)

    casas_preds = _generate_casas_predictions(n_casas, rng, sensor_map, base_date)
    sphere_preds = _generate_sphere_predictions(n_sphere, rng, sensor_map, base_date)

    casas_preds, sphere_preds, injected = _inject_cross_source_anomalies(
        casas_preds, sphere_preds, rng, n_anomalies=n_anomalies
    )

    gc.collect()

    # 1. Single-source: CASAS only
    t0 = time.perf_counter()
    g_casas = build_kg_from_predictions(casas_preds, sensor_map, "casas")
    casas_build_time = time.perf_counter() - t0
    casas_triples = len(g_casas)
    casas_queries = _run_queries(g_casas, SINGLE_SOURCE_QUERIES)
    casas_anomalies = _detect_anomalies_single(g_casas, "casas")

    # 2. Single-source: SPHERE only
    t0 = time.perf_counter()
    g_sphere = build_kg_from_predictions(sphere_preds, sensor_map, "sphere")
    sphere_build_time = time.perf_counter() - t0
    sphere_triples = len(g_sphere)
    sphere_queries = _run_queries(g_sphere, SINGLE_SOURCE_QUERIES)
    sphere_anomalies = _detect_anomalies_single(g_sphere, "sphere")

    # 3. Federated: merged KG
    t0 = time.perf_counter()
    g_fed = build_federated_kg(casas_preds, sphere_preds, sensor_map)
    fed_build_time = time.perf_counter() - t0
    fed_triples = len(g_fed)
    fed_single_queries = _run_queries(g_fed, SINGLE_SOURCE_QUERIES)
    fed_cross_queries = _run_queries(g_fed, FEDERATED_QUERIES)
    fed_anomalies = _detect_anomalies_federated(g_fed)

    result = {
        "trial": trial,
        "n_casas_preds": len(casas_preds),
        "n_sphere_preds": len(sphere_preds),
        "n_injected_anomalies": len(injected),
        "single_source": {
            "casas": {
                "triples": casas_triples,
                "build_time_s": round(casas_build_time, 4),
                "queries": casas_queries,
                "anomalies_detected": casas_anomalies,
            },
            "sphere": {
                "triples": sphere_triples,
                "build_time_s": round(sphere_build_time, 4),
                "queries": sphere_queries,
                "anomalies_detected": sphere_anomalies,
            },
        },
        "federated": {
            "triples": fed_triples,
            "build_time_s": round(fed_build_time, 4),
            "single_queries": fed_single_queries,
            "cross_queries": fed_cross_queries,
            "anomalies_detected": fed_anomalies,
        },
        "injected_anomalies": injected,
    }

    log.info(
        "Trial %d: CASAS=%d, SPHERE=%d, Fed=%d triples | Anomalies: C=%d S=%d F=%d (injected=%d)",
        trial, casas_triples, sphere_triples, fed_triples,
        casas_anomalies["total"], sphere_anomalies["total"],
        fed_anomalies["total"], len(injected),
    )
    return result


def _aggregate_trials(trials, args):
    """Aggregate results across trials."""

    def _avg_anomalies(key_path):
        all_counts = {}
        for t in trials:
            obj = t
            for k in key_path.split("."):
                obj = obj[k]
            for metric, val in obj.items():
                all_counts.setdefault(metric, []).append(float(val))
        return {k: round(float(np.mean(v)), 2) for k, v in all_counts.items()}

    def _avg_queries(key_path):
        all_latencies = {}
        all_counts = {}
        for t in trials:
            obj = t
            for k in key_path.split("."):
                obj = obj[k]
            for qname, qdata in obj.items():
                all_latencies.setdefault(qname, []).append(qdata["avg_ms"])
                all_counts.setdefault(qname, []).append(qdata["result_count"])
        result = {}
        for qname in all_latencies:
            result[qname] = {
                "avg_ms": round(float(np.mean(all_latencies[qname])), 3),
                "result_count_mean": round(float(np.mean(all_counts[qname])), 1),
            }
        return result

    def _avg_scalar(path):
        vals = []
        for t in trials:
            obj = t
            for k in path.split("."):
                obj = obj[k]
            vals.append(float(obj))
        return round(float(np.mean(vals)), 2)

    return {
        "metadata": {
            "n_casas_preds": args.n_casas,
            "n_sphere_preds": args.n_sphere,
            "n_anomalies_injected": args.n_anomalies,
            "trials": args.trials,
            "seed": args.seed,
        },
        "single_source": {
            "casas": {
                "triples_mean": _avg_scalar("single_source.casas.triples"),
                "build_time_mean": _avg_scalar("single_source.casas.build_time_s"),
                "queries": _avg_queries("single_source.casas.queries"),
                "anomalies": _avg_anomalies("single_source.casas.anomalies_detected"),
            },
            "sphere": {
                "triples_mean": _avg_scalar("single_source.sphere.triples"),
                "build_time_mean": _avg_scalar("single_source.sphere.build_time_s"),
                "queries": _avg_queries("single_source.sphere.queries"),
                "anomalies": _avg_anomalies("single_source.sphere.anomalies_detected"),
            },
        },
        "federated": {
            "triples_mean": _avg_scalar("federated.triples"),
            "build_time_mean": _avg_scalar("federated.build_time_s"),
            "single_queries": _avg_queries("federated.single_queries"),
            "cross_queries": _avg_queries("federated.cross_queries"),
            "anomalies": _avg_anomalies("federated.anomalies_detected"),
        },
        "raw_trials": trials,
    }


def _print_summary(summary):
    """Print a human-readable summary."""
    print()
    print("=" * 90)
    print("  CROSS-DATASET FEDERATED REASONING EXPERIMENT")
    print("=" * 90)

    meta = summary["metadata"]
    print(f"  Config: {meta['n_casas_preds']} CASAS + {meta['n_sphere_preds']} SPHERE preds, "
          f"{meta['n_anomalies_injected']} injected anomalies, {meta['trials']} trials")

    print(f"  KG Size (triples):")
    print(f"    CASAS only:   {summary['single_source']['casas']['triples_mean']:.0f}")
    print(f"    SPHERE only:  {summary['single_source']['sphere']['triples_mean']:.0f}")
    print(f"    Federated:    {summary['federated']['triples_mean']:.0f}")

    ca = summary["single_source"]["casas"]["anomalies"]
    sa = summary["single_source"]["sphere"]["anomalies"]
    fa = summary["federated"]["anomalies"]

    hdr = f"    {'Anomaly Type':<42} {'CASAS':>8} {'SPHERE':>8} {'Federated':>10}"
    print(f"  Anomaly Detection:")
    print(hdr)
    print(f"    {'-'*70}")
    for key in ["low_confidence", "night_time_events", "multi_room_motion"]:
        label = key.replace("_", " ").title()
        print(f"    {label:<42} {ca[key]:>8.1f} {sa[key]:>8.1f} {fa[key]:>10.1f}")
    for key in ["cross_source_temporal_overlap", "cross_source_activity_posture_conflict", "cross_source_location_conflict"]:
        label = key.replace("_", " ").title()
        cv = fa.get(key, 0)
        print(f"    {label:<42} {'---':>8} {'---':>8} {cv:>10.1f}")
    print(f"    {'-'*70}")
    print(f"    {'TOTAL':<42} {ca['total']:>8.1f} {sa['total']:>8.1f} {fa['total']:>10.1f}")

    cross_only = fa.get("cross_source_total", 0)
    single_total = ca["total"] + sa["total"]
    improvement = ((fa["total"] - single_total) / single_total * 100) if single_total > 0 else 0
    print(f"    Cross-source anomalies (federated-only): {cross_only:.0f}")
    print(f"    Detection improvement: +{improvement:.1f}% over union of single-source")

    print(f"  Federated Query Latency (cross-source):")
    for qname, qdata in summary["federated"]["cross_queries"].items():
        print(f"    {qname:<42} {qdata['avg_ms']:>8.2f} ms  ({qdata['result_count_mean']:.0f} results)")

    print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Federated Reasoning: single-source vs. federated anomaly detection.",
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--n-casas", type=int, default=100, help="CASAS predictions")
    parser.add_argument("--n-sphere", type=int, default=100, help="SPHERE predictions")
    parser.add_argument("--n-anomalies", type=int, default=10, help="Injected cross-source anomalies")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--outdir", default="outputs/experiments/cross_dataset_federation", help="Output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

    all_trials = []
    for trial in range(args.trials):
        log.info("=" * 60)
        log.info("TRIAL %d / %d", trial + 1, args.trials)
        log.info("=" * 60)
        result = run_federation_trial(
            sensor_map=sensor_map,
            n_casas=args.n_casas,
            n_sphere=args.n_sphere,
            n_anomalies=args.n_anomalies,
            trial=trial,
            seed=args.seed,
        )
        all_trials.append(result)

    summary = _aggregate_trials(all_trials, args)

    out_file = out_dir / "cross_dataset_federation_results.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Results saved to %s", out_file)

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
