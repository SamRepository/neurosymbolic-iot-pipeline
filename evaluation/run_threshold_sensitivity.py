"""
Confidence Threshold Sensitivity Analysis (Enhancement D)
=========================================================
Vary the SWRL confidence thresholds systematically and measure
impact on: (i) FP rate, (ii) correctness, (iii) FeedbackRequired
trigger count, and (iv) F1-weighted score.

The SWRL rules use hardcoded thresholds:
  - 0.85 for validation (Rules 3, 5)
  - 0.80 for posture validation (Rule 4)
  - 0.70 for feedback/hallucination detection (Rule 8)

This experiment sweeps a unified threshold from 0.50 to 0.95 and
measures how each metric responds, identifying the optimal operating
point and quantifying the sensitivity of the pipeline to this parameter.

Usage (Bash):
  PYTHONPATH=. python evaluation/run_threshold_sensitivity.py \
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


# Activity labels per dataset
CASAS_ACTIVITIES = [
    "Meal_Preparation", "Eating", "Wash_Dishes", "Sleeping",
    "Bathing", "Housekeeping", "Bed_to_Toilet", "Enter_Home",
    "Leave_Home", "Personal_Hygiene",
]

SPHERE_ACTIVITIES = [
    "sitting", "standing", "walking", "lying",
    "cooking", "washing_hands", "eating", "sleeping",
]

# Thresholds to sweep
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _gen_predictions(
    n: int,
    dataset: str,
    fp_rate: float = 0.20,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Generate n synthetic predictions with controlled FP rate.

    Predictions span a wide confidence range (0.40-0.99) to ensure
    every threshold in the sweep has meaningful separation.
    """
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
            # FPs cluster at lower confidence but span a range
            confidence = rng.uniform(0.40, 0.82)
        else:
            pred_label = true_label
            # True positives span a wide confidence range
            confidence = rng.uniform(0.50, 0.99)

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


def _evaluate_at_threshold(
    predictions: List[Dict[str, Any]],
    validation_threshold: float,
    feedback_threshold: float,
) -> Dict[str, Any]:
    """Simulate SWRL rule outcomes at a given threshold configuration.

    Mirrors the symbolic reasoner logic:
    - Predictions with confidence >= validation_threshold are ValidatedEvents
    - Predictions with confidence < feedback_threshold are FeedbackRequired
    - Predictions in between are unresolved (neither validated nor flagged)

    Returns metrics: FP rate, correctness, F1-weighted, feedback trigger count.
    """
    from sklearn.metrics import f1_score

    validated = []
    feedback_required = []
    unresolved = []

    for p in predictions:
        conf = p["confidence"]
        if conf >= validation_threshold:
            validated.append(p)
        elif conf < feedback_threshold:
            feedback_required.append(p)
        else:
            unresolved.append(p)

    # Among validated: how many are actually wrong?
    val_fp = sum(1 for p in validated if p["is_false_positive"])
    val_correct = sum(1 for p in validated if not p["is_false_positive"])
    fp_rate = val_fp / max(len(validated), 1)
    correctness = val_correct / max(len(validated), 1)

    # Among feedback_required: how many are true FPs caught?
    feedback_true_catches = sum(1 for p in feedback_required if p["is_false_positive"])

    # F1 on validated predictions only (the ones the system surfaces)
    if validated:
        y_true = [p["ground_truth_label"] for p in validated]
        y_pred = [p["predicted_label"] for p in validated]
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:
        f1 = 0.0

    # Coverage: fraction of total predictions that get validated
    coverage = len(validated) / max(len(predictions), 1)

    return {
        "fp_rate": round(fp_rate, 4),
        "correctness": round(correctness, 4),
        "f1_weighted": round(f1, 4),
        "feedback_triggers": len(feedback_required),
        "feedback_true_catches": feedback_true_catches,
        "validated_count": len(validated),
        "unresolved_count": len(unresolved),
        "coverage": round(coverage, 4),
        "total_predictions": len(predictions),
    }


def run_sensitivity_trial(
    dataset: str,
    n_preds: int,
    thresholds: List[float],
    seed: int,
) -> Dict[str, Any]:
    """Run one trial: sweep all thresholds, collect metrics at each."""
    rng = random.Random(seed)
    predictions = _gen_predictions(n_preds, dataset, fp_rate=0.20, rng=rng)

    threshold_results: List[Dict[str, Any]] = []

    for thresh in thresholds:
        # Validation threshold = thresh
        # Feedback threshold = thresh - 0.15 (maintains paper gap: 0.85/0.70)
        # This mirrors the SWRL rule structure:
        #   Rules 3,5: greaterThan(?conf, validation_threshold) -> Validated
        #   Rule 8: lessThan(?conf, feedback_threshold) -> FeedbackRequired
        feedback_thresh = max(0.30, thresh - 0.15)

        metrics = _evaluate_at_threshold(predictions, thresh, feedback_thresh)
        metrics["threshold"] = thresh
        metrics["feedback_threshold"] = round(feedback_thresh, 2)
        threshold_results.append(metrics)

        log.info(
            "[%s] thresh=%.2f  FP=%.3f  correct=%.3f  F1=%.3f  feedback=%d  coverage=%.3f",
            dataset, thresh, metrics["fp_rate"], metrics["correctness"],
            metrics["f1_weighted"], metrics["feedback_triggers"],
            metrics["coverage"],
        )

    return {
        "dataset": dataset,
        "seed": seed,
        "n_predictions": n_preds,
        "results": threshold_results,
    }


def _aggregate_trials(
    trials: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-threshold metrics across trials (mean +/- std)."""
    from collections import defaultdict

    by_thresh: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        for r in trial["results"]:
            by_thresh[r["threshold"]].append(r)

    aggregated = []
    for thresh in sorted(by_thresh.keys()):
        entries = by_thresh[thresh]
        agg: Dict[str, Any] = {"threshold": thresh}
        for key in ("fp_rate", "correctness", "f1_weighted",
                     "feedback_triggers", "feedback_true_catches",
                     "validated_count", "coverage"):
            vals = [e[key] for e in entries]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
        aggregated.append(agg)

    return aggregated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confidence threshold sensitivity analysis for SWRL rules.",
    )
    p.add_argument("--config", type=str, default="config/base.yaml")
    p.add_argument("--n-preds", type=int, default=300, help="Predictions per trial")
    p.add_argument("--trials", type=int, default=3, help="Trials per dataset")
    p.add_argument("--thresholds", type=str, default=None,
                    help="Comma-separated thresholds (default: 0.50-0.95 in steps of 0.05)")
    p.add_argument("--outdir", type=str,
                    default="outputs/experiments/threshold_sensitivity")
    return p.parse_args()


def main() -> int:
    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging
    from neurosymbolic_iot.utils.seed import set_global_seed

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_global_seed(cfg.get("seed", 42))

    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = DEFAULT_THRESHOLDS

    log.info("=== Confidence Threshold Sensitivity Analysis ===")
    log.info("Thresholds: %s", thresholds)
    log.info("Predictions per trial: %d", args.n_preds)
    log.info("Trials: %d", args.trials)

    datasets = ["casas", "sphere"]
    all_results: Dict[str, Any] = {"datasets": {}, "metadata": {}}

    for ds in datasets:
        log.info("--- Dataset: %s ---", ds.upper())
        ds_trials = []
        for trial_idx in range(args.trials):
            seed = cfg.get("seed", 42) + trial_idx * 1000
            log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, args.trials, seed)
            trial_result = run_sensitivity_trial(
                dataset=ds,
                n_preds=args.n_preds,
                thresholds=thresholds,
                seed=seed,
            )
            ds_trials.append(trial_result)

        aggregated = _aggregate_trials(ds_trials)
        all_results["datasets"][ds] = {
            "aggregated": aggregated,
            "raw_trials": ds_trials,
        }

        # Log summary
        for agg in aggregated:
            log.info(
                "  [%s] thresh=%.2f  FP=%.3f+/-%.3f  correct=%.3f+/-%.3f  F1=%.3f+/-%.3f  feedback=%.0f+/-%.0f",
                ds, agg["threshold"],
                agg["fp_rate_mean"], agg["fp_rate_std"],
                agg["correctness_mean"], agg["correctness_std"],
                agg["f1_weighted_mean"], agg["f1_weighted_std"],
                agg["feedback_triggers_mean"], agg["feedback_triggers_std"],
            )

    all_results["metadata"] = {
        "thresholds": thresholds,
        "n_predictions": args.n_preds,
        "trials": args.trials,
        "fp_injection_rate": 0.20,
        "timestamp": datetime.now().isoformat(),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "threshold_sensitivity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

