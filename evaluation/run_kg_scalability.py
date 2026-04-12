"""
KG Scalability Experiment (Section 4.2 — Figure 4)
===================================================
Benchmarks SPARQL query latency, KG build throughput, and peak memory
as the knowledge graph grows from 500 to 20,000 triples.

Generates synthetic but realistic RDF triples using the production
KG builder (Algorithm 1) and measures five representative SPARQL
queries at each scale point.

Usage (Bash):
  PYTHONPATH=. python evaluation/run_kg_scalability.py --config config/base.yaml --trials 3

Usage (PowerShell):
  $env:PYTHONPATH="."; python evaluation/run_kg_scalability.py --config config/base.yaml --trials 3
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

from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
    build_kg_from_predictions,
    load_sensor_map,
)
from evaluation.metrics_collector import LatencyTracker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespaces used in SPARQL queries (must match kg_federation_loader.py)
# ---------------------------------------------------------------------------
NSIOT = "http://example.org/neuro-symbolic-iot#"
TIME = "http://www.w3.org/2006/time#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

DEFAULT_SIZES = [500, 1000, 1500, 2000, 2500, 3000, 5000, 10000, 20000]
QUERIES_PER_RUN = 20  # repetitions of each query per trial

# ---------------------------------------------------------------------------
# SPARQL query suite — five representative queries
# ---------------------------------------------------------------------------
SPARQL_QUERIES: Dict[str, str] = {
    "Q1_count_predictions": f"""
        SELECT ?pred WHERE {{
            ?pred <{RDF}type> <{NSIOT}NeuralPrediction> .
        }}
    """,
    "Q2_events_with_person": f"""
        SELECT ?e ?p WHERE {{
            ?e <{RDF}type> <{NSIOT}Event> .
            ?e <{NSIOT}involvesPerson> ?p .
        }}
    """,
    "Q3_high_confidence": f"""
        SELECT ?pred ?conf WHERE {{
            ?pred <{NSIOT}hasConfidenceScore> ?conf .
            FILTER(?conf > 0.85)
        }}
    """,
    "Q4_sensor_locations": f"""
        SELECT ?s ?room WHERE {{
            ?s <{NSIOT}isLocatedIn> ?room .
            ?s <{RDF}type> <{NSIOT}PIRMotionSensor> .
        }}
    """,
    "Q5_temporal_join": f"""
        SELECT ?e ?time WHERE {{
            ?e <{NSIOT}hasTemporalEntity> ?t .
            ?t <{TIME}inXSDDateTimeStamp> ?time .
        }}
    """,
}


# ---------------------------------------------------------------------------
# Synthetic prediction generation
# ---------------------------------------------------------------------------

_CASAS_SENSORS = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08",
    "D01", "D02", "D03", "AD1-A", "AD1-B", "AD1-C",
]
_CASAS_STATES = {"M": ["ON", "OFF"], "D": ["OPEN", "CLOSE"], "A": ["ON", "OFF"]}

_SPHERE_PIRS = ["bath", "bed1", "bed2", "hall", "kitchen", "living", "stairs", "study", "toilet"]


def _random_timestamp(rng: random.Random) -> str:
    """Generate a random ISO timestamp in 2024."""
    base = datetime(2024, 1, 1)
    offset = timedelta(seconds=rng.randint(0, 365 * 24 * 3600))
    return (base + offset).isoformat()


def _generate_casas_prediction(rng: random.Random, idx: int, activities: List[str]) -> Dict[str, Any]:
    """Generate one synthetic CASAS prediction dict."""
    n_sensors = rng.randint(3, 6)
    sensors = rng.sample(_CASAS_SENSORS, min(n_sensors, len(_CASAS_SENSORS)))
    tokens = []
    for s in sensors:
        prefix = s[0]
        states = _CASAS_STATES.get(prefix, ["ON", "OFF"])
        tokens.append(f"{s}:{rng.choice(states)}")

    return {
        "predicted_label": rng.choice(activities),
        "confidence": round(rng.uniform(0.50, 0.99), 3),
        "window_start": _random_timestamp(rng),
        "window_end": _random_timestamp(rng),
        "metadata": {
            "sensor_tokens": tokens,
            "model_tag": "gru_scalability_test",
        },
    }


def _generate_sphere_prediction(rng: random.Random, idx: int, activities: List[str]) -> Dict[str, Any]:
    """Generate one synthetic SPHERE prediction dict."""
    n_pirs = rng.randint(2, 4)
    pirs = rng.sample(_SPHERE_PIRS, min(n_pirs, len(_SPHERE_PIRS)))

    return {
        "predicted_label": rng.choice(activities),
        "confidence": round(rng.uniform(0.50, 0.99), 3),
        "window_start": _random_timestamp(rng),
        "window_end": _random_timestamp(rng),
        "metadata": {
            "active_pirs": pirs,
            "model_tag": "lstm_scalability_test",
        },
    }


def _generate_synthetic_predictions(
    n: int,
    dataset: str,
    sensor_map: Dict[str, Any],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate n synthetic predictions for the given dataset."""
    rng = random.Random(seed)
    ds_map = sensor_map.get(dataset, {})
    activities = list(ds_map.get("activity_map", {"default": "Activity"}).keys())

    if dataset == "casas":
        return [_generate_casas_prediction(rng, i, activities) for i in range(n)]
    else:
        return [_generate_sphere_prediction(rng, i, activities) for i in range(n)]


def _calibrate_triples_per_prediction(
    dataset: str,
    sensor_map: Dict[str, Any],
    seed: int = 42,
) -> float:
    """Measure average triples per prediction for this dataset."""
    preds = _generate_synthetic_predictions(20, dataset, sensor_map, seed=seed)
    g = build_kg_from_predictions(preds, sensor_map, dataset)
    # Subtract 1 for the Person triple (shared, not per-prediction)
    return (len(g) - 1) / len(preds)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_sparql_benchmark(
    graph: Any,
    queries_per_run: int = QUERIES_PER_RUN,
) -> Dict[str, Dict[str, float]]:
    """Run all SPARQL queries on the graph and return latency stats."""
    results: Dict[str, Dict[str, float]] = {}
    for qname, sparql in SPARQL_QUERIES.items():
        tracker = LatencyTracker(name=qname)
        for _ in range(queries_per_run):
            t0 = time.perf_counter()
            list(graph.query(sparql))  # materialize results
            tracker.record(time.perf_counter() - t0)
        results[qname] = tracker.summary()
    return results


def benchmark_kg_size(
    target_triples: int,
    dataset: str,
    sensor_map: Dict[str, Any],
    triples_per_pred: float,
    trial: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a KG at the target size and benchmark it."""
    n_preds = max(1, int(round(target_triples / triples_per_pred)))
    trial_seed = seed + trial * 1000 + target_triples

    gc.collect()

    try:
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss
    except ImportError:
        process = None
        mem_before = 0

    # Build KG
    preds = _generate_synthetic_predictions(n_preds, dataset, sensor_map, seed=trial_seed)
    t0 = time.perf_counter()
    graph = build_kg_from_predictions(preds, sensor_map, dataset)
    build_time = time.perf_counter() - t0

    actual_triples = len(graph)
    throughput = actual_triples / max(build_time, 1e-6)

    if process:
        peak_memory_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    else:
        peak_memory_mb = 0.0

    # Run SPARQL queries
    query_results = _run_sparql_benchmark(graph)

    # Aggregate query latency across all queries
    all_avgs = [v["avg_ms"] for v in query_results.values()]
    all_p95s = [v["p95_ms"] for v in query_results.values()]

    result = {
        "target_triples": target_triples,
        "actual_triples": actual_triples,
        "n_predictions": n_preds,
        "build_time_s": round(build_time, 4),
        "throughput_triples_per_sec": round(throughput, 1),
        "avg_query_latency_ms": round(float(np.mean(all_avgs)), 3),
        "p95_query_latency_ms": round(float(np.mean(all_p95s)), 3),
        "peak_memory_mb": peak_memory_mb,
        "per_query": query_results,
        "trial": trial,
        "dataset": dataset,
    }

    log.info(
        "  %s trial=%d | target=%d actual=%d | build=%.3fs | latency=%.2fms | throughput=%.0f t/s | mem=%.0fMB",
        dataset, trial, target_triples, actual_triples,
        build_time, result["avg_query_latency_ms"], throughput, peak_memory_mb,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KG Scalability: benchmark SPARQL latency, throughput, memory vs. KG size.",
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated target triple counts (default: 500,...,20000)",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per (size, dataset) pair")
    parser.add_argument("--outdir", default="outputs/experiments/kg_scalability", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging
    from neurosymbolic_iot.utils.seed import set_global_seed

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_global_seed(args.seed)

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sensor map
    kg_cfg = cfg.get("kg", {})
    sensor_map_path = Path(
        kg_cfg.get("sensor_map_path", "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json")
    )
    sensor_map = load_sensor_map(sensor_map_path)

    datasets = ["casas", "sphere"]

    # Calibrate triples per prediction
    tpp: Dict[str, float] = {}
    for ds in datasets:
        tpp[ds] = _calibrate_triples_per_prediction(ds, sensor_map, seed=args.seed)
        log.info("Calibrated %s: %.1f triples/prediction", ds, tpp[ds])

    # Run benchmarks
    all_results: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in datasets}

    for ds in datasets:
        log.info("=" * 60)
        log.info("Benchmarking %s (%d sizes x %d trials)", ds.upper(), len(sizes), args.trials)
        log.info("=" * 60)
        for size in sizes:
            for trial in range(args.trials):
                result = benchmark_kg_size(
                    target_triples=size,
                    dataset=ds,
                    sensor_map=sensor_map,
                    triples_per_pred=tpp[ds],
                    trial=trial,
                    seed=args.seed,
                )
                all_results[ds].append(result)

    # Aggregate: mean ± std across trials per size
    summary: Dict[str, Any] = {"datasets": {}, "metadata": {
        "sizes": sizes,
        "trials": args.trials,
        "seed": args.seed,
        "triples_per_prediction": tpp,
    }}

    for ds in datasets:
        ds_summary: List[Dict[str, Any]] = []
        for size in sizes:
            trials = [r for r in all_results[ds] if r["target_triples"] == size]
            agg = {
                "target_triples": size,
                "actual_triples_mean": round(float(np.mean([t["actual_triples"] for t in trials])), 0),
                "build_time_mean": round(float(np.mean([t["build_time_s"] for t in trials])), 4),
                "build_time_std": round(float(np.std([t["build_time_s"] for t in trials])), 4),
                "throughput_mean": round(float(np.mean([t["throughput_triples_per_sec"] for t in trials])), 1),
                "throughput_std": round(float(np.std([t["throughput_triples_per_sec"] for t in trials])), 1),
                "avg_query_latency_mean": round(float(np.mean([t["avg_query_latency_ms"] for t in trials])), 3),
                "avg_query_latency_std": round(float(np.std([t["avg_query_latency_ms"] for t in trials])), 3),
                "p95_query_latency_mean": round(float(np.mean([t["p95_query_latency_ms"] for t in trials])), 3),
                "peak_memory_mean": round(float(np.mean([t["peak_memory_mb"] for t in trials])), 1),
                "peak_memory_std": round(float(np.std([t["peak_memory_mb"] for t in trials])), 1),
            }
            ds_summary.append(agg)
        summary["datasets"][ds] = ds_summary

    # Save
    out_file = out_dir / "kg_scalability_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Results saved to %s", out_file)

    # Print summary table
    print("\n" + "=" * 100)
    print("  KG SCALABILITY EXPERIMENT — SPARQL Query Latency vs. KG Size")
    print("=" * 100)

    for ds in datasets:
        print(f"\n  {ds.upper()}")
        print(f"  {'Triples':>8} {'Actual':>8} {'Build(s)':>10} {'Throughput':>14} {'Latency(ms)':>16} {'Memory(MB)':>12}")
        print("  " + "-" * 72)
        for row in summary["datasets"][ds]:
            lat = f"{row['avg_query_latency_mean']:.2f} +/- {row['avg_query_latency_std']:.2f}"
            thr = f"{row['throughput_mean']:.0f} +/- {row['throughput_std']:.0f}"
            mem = f"{row['peak_memory_mean']:.0f} +/- {row['peak_memory_std']:.0f}"
            print(
                f"  {row['target_triples']:>8} {row['actual_triples_mean']:>8.0f} "
                f"{row['build_time_mean']:>10.4f} {thr:>14} {lat:>16} {mem:>12}"
            )

    print("\n" + "=" * 100)
    print(f"Results saved to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
