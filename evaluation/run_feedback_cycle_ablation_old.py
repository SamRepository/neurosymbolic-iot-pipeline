"""
Feedback Cycle Ablation Experiment (Enhancement H)
===================================================
Track F1-weighted, FP rate, and correctness at each feedback cycle (0-8)
for both CASAS and SPHERE datasets.

Usage (Bash):
  PYTHONPATH=. python evaluation/run_feedback_cycle_ablation.py
      --config config/base.yaml --trials 3
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


CASAS_ACTIVITIES = [
    "Meal_Preparation", "Eating", "Wash_Dishes", "Sleeping",
    "Bathing", "Housekeeping", "Bed_to_Toilet", "Enter_Home",
    "Leave_Home", "Personal_Hygiene",
]

SPHERE_ACTIVITIES = [
    "sitting", "standing", "walking", "lying",
    "cooking", "washing_hands", "eating", "sleeping",
]


def _gen_predictions(
    n: int,
    dataset: str,
    fp_rate: float = 0.20,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Generate n synthetic predictions with a controlled false-positive rate."""
    if rng is None:
        rng = random.Random(42)

    activities = CASAS_ACTIVITIES if dataset == "casas" else SPHERE_ACTIVITIES
    base_time = datetime(2024, 3, 1, 8, 0, 0)

    preds: List[Dict[str, Any]] = []
    for i in range(n):
        true_label = rng.choice(activities)
        is_fp = rng.random() < fp_rate
        if is_fp:
            wrong_choices = [a for a in activities if a != true_label]
            pred_label = rng.choice(wrong_choices)
            confidence = rng.uniform(0.55, 0.80)
        else:
            pred_label = true_label
            confidence = rng.uniform(0.70, 0.99)

        t_start = base_time + timedelta(minutes=i * 5)
        t_end = t_start + timedelta(minutes=rng.randint(2, 10))

        preds.append({
            "sample_idx": i,
            "predicted_label": pred_label,
            "ground_truth_label": true_label,
            "confidence": round(confidence, 4),
            "is_false_positive": is_fp,
            "window_start": t_start.isoformat(),
            "window_end": t_end.isoformat(),
            "metadata": {
                "event_uri": f"http://example.org/neuro-symbolic-iot#event_{dataset}_{i}",
            },
        })
    return preds


def _compute_metrics(
    predictions: List[Dict[str, Any]],
    conf_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1-weighted, FP rate, and correctness from predictions."""
    from sklearn.metrics import f1_score

    active = [p for p in predictions if p["confidence"] >= conf_threshold]
    if not active:
        return {"f1_weighted": 0.0, "fp_rate": 0.0, "correctness": 0.0}

    y_true = [p["ground_truth_label"] for p in active]
    y_pred = [p["predicted_label"] for p in active]

    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # FP rate = fraction of active predictions that are wrong
    fp_count = sum(1 for p in active if p["is_false_positive"])
    fp_rate = fp_count / len(active)

    # Correctness = fraction matching ground truth
    correct_count = sum(1 for p in active if p["predicted_label"] == p["ground_truth_label"])
    correctness = correct_count / len(active)

    return {
        "f1_weighted": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
        "correctness": round(correctness, 4),
        "active_predictions": len(active),
        "total_predictions": len(predictions),
    }


def _simulate_one_feedback_cycle(
    predictions: List[Dict[str, Any]],
    conf_threshold: float,
    adjustment_rate: float = 0.05,
    rng: random.Random | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Simulate one feedback cycle.

    1. Identify false-positive predictions above threshold.
    2. Reduce detected FP confidence by 15-30%.
    3. Correct some FPs (flip label to ground truth).
    4. Adjust confidence threshold based on FP ratio.
    """
    if rng is None:
        rng = random.Random(42)

    updated = copy.deepcopy(predictions)

    active_fps = [
        p for p in updated
        if p["is_false_positive"] and p["confidence"] >= conf_threshold
    ]

    active_total = sum(1 for p in updated if p["confidence"] >= conf_threshold)
    fp_ratio = len(active_fps) / max(active_total, 1)

    # Penalize detected FPs
    for p in active_fps:
        detect_prob = 0.5 + 0.3 * (1.0 - p["confidence"])
        if rng.random() < detect_prob:
            reduction = rng.uniform(0.15, 0.30)
            p["confidence"] = round(max(0.1, p["confidence"] - reduction), 4)

            # Some detected FPs get corrected (simulates retrain)
            if rng.random() < 0.3:
                p["predicted_label"] = p["ground_truth_label"]
                p["is_false_positive"] = False

    # Threshold adjustment (mirrors adapt_kg logic)
    if fp_ratio > 0.3:
        conf_threshold = min(0.99, conf_threshold + adjustment_rate)
    elif fp_ratio < 0.1:
        conf_threshold = max(0.5, conf_threshold - adjustment_rate * 0.5)

    return updated, round(conf_threshold, 4)


def run_ablation_trial(
    dataset: str,
    n_preds: int,
    max_cycles: int,
    seed: int,
    initial_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run one ablation trial: cycles 0 through max_cycles."""
    rng = random.Random(seed)

    predictions = _gen_predictions(n_preds, dataset, fp_rate=0.20, rng=rng)

    conf_threshold = initial_threshold
    cycle_metrics: List[Dict[str, Any]] = []

    for cycle in range(max_cycles + 1):
        metrics = _compute_metrics(predictions, conf_threshold)
        metrics["cycle"] = cycle
        metrics["conf_threshold"] = conf_threshold
        cycle_metrics.append(metrics)

        log.info(
            "[%s] cycle=%d  F1=%.3f  FP=%.3f  correctness=%.3f  threshold=%.3f",
            dataset, cycle, metrics["f1_weighted"], metrics["fp_rate"],
            metrics["correctness"], conf_threshold,
        )

        if cycle < max_cycles:
            predictions, conf_threshold = _simulate_one_feedback_cycle(
                predictions, conf_threshold, adjustment_rate=0.05, rng=rng,
            )

    return {
        "dataset": dataset,
        "seed": seed,
        "n_predictions": n_preds,
        "cycles": cycle_metrics,
    }


def _aggregate_trials(
    trials: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-cycle metrics across trials (mean +/- std)."""
    from collections import defaultdict

    by_cycle: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for trial in trials:
        for cm in trial["cycles"]:
            by_cycle[cm["cycle"]].append(cm)

    aggregated = []
    for cycle_id in sorted(by_cycle.keys()):
        entries = by_cycle[cycle_id]
        agg: Dict[str, Any] = {"cycle": cycle_id}
        for key in ("f1_weighted", "fp_rate", "correctness", "conf_threshold"):
            vals = [e[key] for e in entries]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
        aggregated.append(agg)

    return aggregated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feedback cycle ablation: track F1, FP rate, correctness over cycles 0-8.",
    )
    p.add_argument("--config", type=str, default="config/base.yaml", help="Base config path")
    p.add_argument("--n-preds", type=int, default=200, help="Predictions per trial")
    p.add_argument("--max-cycles", type=int, default=8, help="Max feedback cycles")
    p.add_argument("--trials", type=int, default=3, help="Number of trials per dataset")
    p.add_argument("--outdir", type=str, default="outputs/experiments/feedback_cycle_ablation",
                    help="Output directory for results JSON")
    return p.parse_args()


def main() -> int:
    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging
    from neurosymbolic_iot.utils.seed import set_global_seed

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_global_seed(cfg.get("seed", 42))

    log.info("=== Feedback Cycle Ablation Experiment ===")
    log.info("Predictions per trial: %d", args.n_preds)
    log.info("Max cycles: %d", args.max_cycles)
    log.info("Trials: %d", args.trials)

    datasets = ["casas", "sphere"]
    all_results: Dict[str, Any] = {"datasets": {}, "metadata": {}}

    for ds in datasets:
        log.info("--- Dataset: %s ---", ds.upper())
        ds_trials = []
        for trial_idx in range(args.trials):
            seed = cfg.get("seed", 42) + trial_idx * 1000
            log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, args.trials, seed)
            trial_result = run_ablation_trial(
                dataset=ds,
                n_preds=args.n_preds,
                max_cycles=args.max_cycles,
                seed=seed,
            )
            ds_trials.append(trial_result)

        aggregated = _aggregate_trials(ds_trials)
        all_results["datasets"][ds] = {
            "aggregated": aggregated,
            "raw_trials": ds_trials,
        }

        for agg in aggregated:
            log.info(
                "  [%s] cycle=%d  F1=%.3f+/-%.3f  FP=%.3f+/-%.3f  correct=%.3f+/-%.3f",
                ds, agg["cycle"],
                agg["f1_weighted_mean"], agg["f1_weighted_std"],
                agg["fp_rate_mean"], agg["fp_rate_std"],
                agg["correctness_mean"], agg["correctness_std"],
            )

    all_results["metadata"] = {
        "n_predictions": args.n_preds,
        "max_cycles": args.max_cycles,
        "trials": args.trials,
        "fp_injection_rate": 0.20,
        "timestamp": datetime.now().isoformat(),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "feedback_cycle_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

