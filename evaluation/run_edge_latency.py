"""
Edge Deployment Latency Benchmark (Enhancement G)
==================================================
CPU-only end-to-end latency per window through all 4 pipeline stages:
  1. Neural inference (GRU for CASAS, LSTM for SPHERE)
  2. KG population (build_kg_from_predictions)
  3. Symbolic reasoning (SPARQL queries on rdflib)
  4. Feedback detection (contradiction detection + retraction)

Varies batch sizes (1, 5, 10, 20, 50 windows) to show how latency scales
on a simulated edge node (CPU-only, no GPU).

Grounds Algorithm 3 (Edge-Fog-Cloud Task Orchestration) from Section 3.5.

Usage (Bash):
  PYTHONPATH=. python evaluation/run_edge_latency.py --config config/base.yaml --trials 5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

log = logging.getLogger(__name__)

CASAS_ACTIVITIES = [
    "Cook", "Eat", "Clean", "PhoneCall", "WashHands",
]
CASAS_SENSORS = [
    "M01:ON", "M02:OFF", "M03:ON", "M04:OFF", "M05:ON",
    "M06:OFF", "M07:ON", "M08:OFF", "D01:OPEN", "D02:CLOSE",
    "D03:OPEN", "AD1-A:ON", "AD1-B:OFF", "AD1-C:ON",
]

SPHERE_ACTIVITIES = [
    "a_walk", "p_sit", "p_stand", "p_lie", "t_sit_stand",
    "t_stand_sit", "a_ascend", "a_descend", "p_bent", "t_bend",
]
SPHERE_PIRS = ["bath", "bed1", "bed2", "hall", "kitchen", "living", "stairs", "study", "toilet"]


def _gen_casas_predictions(
    n: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    base_time = datetime(2024, 6, 15, 8, 0, 0)
    preds: List[Dict[str, Any]] = []
    max_seq_len = 256

    token_ids = torch.zeros(n, max_seq_len, dtype=torch.long)
    time_deltas = torch.zeros(n, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        activity = rng.choice(CASAS_ACTIVITIES)
        conf = round(rng.uniform(0.55, 0.99), 4)
        t_start = base_time + timedelta(minutes=i * 5)
        t_end = t_start + timedelta(minutes=rng.randint(2, 10))
        n_sensors = rng.randint(3, 6)
        sensors = rng.sample(CASAS_SENSORS, min(n_sensors, len(CASAS_SENSORS)))

        preds.append({
            "predicted_label": activity,
            "ground_truth_label": activity if rng.random() > 0.2 else rng.choice(CASAS_ACTIVITIES),
            "confidence": conf,
            "window_start": t_start.isoformat(),
            "window_end": t_end.isoformat(),
            "metadata": {
                "sensor_tokens": sensors,
                "event_uri": f"http://example.org/neuro-symbolic-iot#event_casas_{i}",
                "model_tag": "casas_gru_edge",
            },
        })

        seq_len = rng.randint(10, max_seq_len)
        lengths[i] = seq_len
        token_ids[i, :seq_len] = torch.randint(1, 200, (seq_len,))
        time_deltas[i, :seq_len] = torch.rand(seq_len) * 60.0

    return preds, token_ids, time_deltas, lengths


def _gen_sphere_predictions(
    n: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    base_time = datetime(2024, 6, 15, 8, 0, 0)
    preds: List[Dict[str, Any]] = []
    seq_len = 128
    in_dim = 3

    sequences = torch.randn(n, seq_len, in_dim)

    for i in range(n):
        activity = rng.choice(SPHERE_ACTIVITIES)
        conf = round(rng.uniform(0.55, 0.99), 4)
        t_start = base_time + timedelta(seconds=i * 30)
        t_end = t_start + timedelta(seconds=30)
        n_pirs = rng.randint(2, 4)
        active_pirs = rng.sample(SPHERE_PIRS, min(n_pirs, len(SPHERE_PIRS)))

        preds.append({
            "predicted_label": activity,
            "ground_truth_label": activity if rng.random() > 0.2 else rng.choice(SPHERE_ACTIVITIES),
            "confidence": conf,
            "window_start": t_start.isoformat(),
            "window_end": t_end.isoformat(),
            "metadata": {
                "active_pirs": active_pirs,
                "event_uri": f"http://example.org/neuro-symbolic-iot#event_sphere_{i}",
                "model_tag": "sphere_lstm_edge",
            },
        })

    return preds, sequences


SPARQL_QUERIES = {
    "Q1_count_predictions": """
        PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
        SELECT (COUNT(?pred) AS ?cnt) WHERE { ?pred a nsiot:NeuralPrediction }
    """,
    "Q2_events_with_person": """
        PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
        SELECT ?e WHERE { ?e a nsiot:Event . ?e nsiot:involvesPerson ?p }
    """,
    "Q3_high_confidence": """
        PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?pred ?conf WHERE {
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf > 0.85)
        }
    """,
    "Q4_sensor_locations": """
        PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
        SELECT ?s ?room WHERE {
            ?s nsiot:isLocatedIn ?room .
            ?s a nsiot:PIRMotionSensor
        }
    """,
    "Q5_temporal_join": """
        PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
        PREFIX time: <http://www.w3.org/2006/time#>
        SELECT ?e ?ts WHERE {
            ?e nsiot:hasTemporalEntity ?t .
            ?t time:inXSDDateTimeStamp ?ts
        }
    """,
}


def _simulate_feedback_detection(
    predictions: List[Dict[str, Any]],
    conf_threshold: float = 0.85,
    feedback_threshold: float = 0.70,
) -> Dict[str, Any]:
    validated = []
    feedback_required = []
    retracted = 0

    for pred in predictions:
        conf = pred["confidence"]
        if conf >= conf_threshold:
            validated.append(pred)
        elif conf < feedback_threshold:
            feedback_required.append(pred)
            if pred.get("predicted_label") != pred.get("ground_truth_label"):
                retracted += 1

    return {
        "validated_count": len(validated),
        "feedback_count": len(feedback_required),
        "retracted_count": retracted,
    }


def benchmark_one_batch(
    dataset: str,
    batch_size: int,
    sensor_map: Dict[str, Any],
    casas_model: torch.nn.Module,
    sphere_model: torch.nn.Module,
    rng: random.Random,
) -> Dict[str, Any]:
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
        build_kg_from_predictions,
    )

    timings: Dict[str, float] = {}
    memory_before = _get_memory_mb()

    # Stage 1: Neural Inference (CPU-only)
    t0 = time.perf_counter()
    if dataset == "casas":
        preds, token_ids, time_deltas, lengths = _gen_casas_predictions(batch_size, rng)
        with torch.no_grad():
            _ = casas_model(token_ids, time_deltas, lengths)
    else:
        preds, sequences = _gen_sphere_predictions(batch_size, rng)
        with torch.no_grad():
            _ = sphere_model(sequences)
    timings["neural_inference_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 2: KG Population
    t0 = time.perf_counter()
    graph = build_kg_from_predictions(preds, sensor_map, dataset)
    timings["kg_population_ms"] = (time.perf_counter() - t0) * 1000.0
    timings["triples_built"] = len(graph)

    # Stage 3: Symbolic Reasoning (SPARQL on rdflib)
    t0 = time.perf_counter()
    query_results: Dict[str, int] = {}
    for qname, sparql in SPARQL_QUERIES.items():
        results = list(graph.query(sparql))
        query_results[qname] = len(results)
    timings["reasoning_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 4: Feedback Detection
    t0 = time.perf_counter()
    fb = _simulate_feedback_detection(preds)
    timings["feedback_ms"] = (time.perf_counter() - t0) * 1000.0

    # Totals
    timings["total_e2e_ms"] = (
        timings["neural_inference_ms"]
        + timings["kg_population_ms"]
        + timings["reasoning_ms"]
        + timings["feedback_ms"]
    )
    timings["per_window_ms"] = timings["total_e2e_ms"] / batch_size
    timings["throughput_windows_per_sec"] = batch_size / (timings["total_e2e_ms"] / 1000.0)

    memory_after = _get_memory_mb()
    timings["memory_delta_mb"] = round(memory_after - memory_before, 1)
    timings["peak_memory_mb"] = round(memory_after, 1)

    timings["batch_size"] = batch_size
    timings["feedback_stats"] = fb
    timings["query_result_counts"] = query_results

    return timings


def _get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_edge_benchmark(
    cfg: Dict[str, Any],
    batch_sizes: List[int],
    trials: int,
    seed: int,
) -> Dict[str, Any]:
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import load_sensor_map
    from neurosymbolic_iot.neural_perception.models import CasasGRUClassifier, SphereLSTMClassifier

    sensor_map_path = Path(cfg.get("kg", {}).get("sensor_map_path", ""))
    sensor_map = load_sensor_map(sensor_map_path)

    np_casas = cfg.get("neural_perception", {}).get("casas", {})
    casas_model = CasasGRUClassifier(
        vocab_size=200,
        num_classes=len(CASAS_ACTIVITIES),
        emb_dim=int(np_casas.get("emb_dim", 64)),
        hidden=int(np_casas.get("hidden_size", 128)),
        num_layers=int(np_casas.get("num_layers", 1)),
        dropout=0.0,
    )
    casas_model.eval()

    np_sphere = cfg.get("neural_perception", {}).get("sphere", {})
    sphere_model = SphereLSTMClassifier(
        in_dim=3,
        num_classes=len(SPHERE_ACTIVITIES),
        hidden=int(np_sphere.get("hidden_size", 128)),
        num_layers=int(np_sphere.get("num_layers", 1)),
        dropout=0.0,
    )
    sphere_model.eval()

    log.info("Models built (CPU-only, eval mode)")
    log.info("CASAS GRU params: %d", sum(p.numel() for p in casas_model.parameters()))
    log.info("SPHERE LSTM params: %d", sum(p.numel() for p in sphere_model.parameters()))

    all_results: Dict[str, Any] = {"datasets": {}, "metadata": {}}

    for dataset in ["casas", "sphere"]:
        log.info("=" * 60)
        log.info("Benchmarking: %s", dataset.upper())
        log.info("=" * 60)

        ds_results: List[Dict[str, Any]] = []

        for bs in batch_sizes:
            log.info("  Batch size: %d", bs)
            trial_timings: List[Dict[str, Any]] = []

            for trial_idx in range(trials):
                rng = random.Random(seed + trial_idx * 1000 + bs)
                timing = benchmark_one_batch(
                    dataset=dataset,
                    batch_size=bs,
                    sensor_map=sensor_map,
                    casas_model=casas_model,
                    sphere_model=sphere_model,
                    rng=rng,
                )
                trial_timings.append(timing)

            agg = _aggregate_timings(trial_timings, bs)
            ds_results.append(agg)

            log.info(
                "    E2E=%.1f+/-%.1fms  per_window=%.1f+/-%.1fms  "
                "neural=%.1fms  kg=%.1fms  reasoning=%.1fms  feedback=%.1fms",
                agg["total_e2e_ms_mean"], agg["total_e2e_ms_std"],
                agg["per_window_ms_mean"], agg["per_window_ms_std"],
                agg["neural_inference_ms_mean"], agg["kg_population_ms_mean"],
                agg["reasoning_ms_mean"], agg["feedback_ms_mean"],
            )

        all_results["datasets"][dataset] = ds_results

    all_results["metadata"] = {
        "batch_sizes": batch_sizes,
        "trials_per_batch": trials,
        "seed": seed,
        "device": "cpu",
        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
        "casas_model_params": sum(p.numel() for p in casas_model.parameters()),
        "sphere_model_params": sum(p.numel() for p in sphere_model.parameters()),
    }

    return all_results


def _aggregate_timings(
    trial_timings: List[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, Any]:
    keys = [
        "neural_inference_ms", "kg_population_ms", "reasoning_ms",
        "feedback_ms", "total_e2e_ms", "per_window_ms",
        "throughput_windows_per_sec", "peak_memory_mb", "triples_built",
    ]
    agg: Dict[str, Any] = {"batch_size": batch_size}
    for k in keys:
        vals = [t[k] for t in trial_timings if k in t]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 2)
            agg[f"{k}_std"] = round(float(np.std(vals)), 2)
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Edge deployment latency benchmark (CPU-only, all 4 pipeline stages).",
    )
    p.add_argument("--config", type=str, default="config/base.yaml", help="Base config path")
    p.add_argument("--batch-sizes", type=str, default="1,5,10,20,50",
                    help="Comma-separated batch sizes (windows per batch)")
    p.add_argument("--trials", type=int, default=5, help="Trials per batch size")
    p.add_argument("--outdir", type=str, default="outputs/experiments/edge_latency",
                    help="Output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    log.info("=== Edge Deployment Latency Benchmark ===")
    log.info("Batch sizes: %s", batch_sizes)
    log.info("Trials per batch: %d", args.trials)
    log.info("Device: CPU-only (simulating edge node)")

    results = run_edge_benchmark(cfg, batch_sizes, args.trials, seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "edge_latency_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", out_path)

    _print_summary(results)

    return 0


def _print_summary(results: Dict[str, Any]) -> None:
    print("")
    print("=" * 90)
    print("  EDGE DEPLOYMENT LATENCY BENCHMARK (CPU-only)")
    print("=" * 90)

    for ds in ["casas", "sphere"]:
        ds_data = results["datasets"].get(ds, [])
        if not ds_data:
            continue

        print(f"  --- {ds.upper()} ---")
        print(f"  {"Batch":>6} {"Neural":>10} {"KG Pop":>10} {"Reason":>10} {"Feedback":>10} {"Total E2E":>12} {"Per-Win":>10} {"Win/s":>8}")
        print("  " + "-" * 84)

        for row in ds_data:
            bs = row["batch_size"]
            neural = row["neural_inference_ms_mean"]
            kg = row["kg_population_ms_mean"]
            reason = row["reasoning_ms_mean"]
            fb = row["feedback_ms_mean"]
            total = row["total_e2e_ms_mean"]
            pw = row["per_window_ms_mean"]
            tput = row["throughput_windows_per_sec_mean"]
            print(f"  {bs:>6} {neural:>8.1f}ms {kg:>8.1f}ms {reason:>8.1f}ms {fb:>8.1f}ms {total:>10.1f}ms {pw:>8.1f}ms {tput:>7.1f}")

    print("=" * 90)


if __name__ == "__main__":
    raise SystemExit(main())
