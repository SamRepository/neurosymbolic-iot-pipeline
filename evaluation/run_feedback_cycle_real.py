"""
Feedback Cycle Real-Data Evaluation
===================================

Held-out, non-simulated counterpart to ``run_feedback_cycle_ablation.py``.

This experiment runs the *real* feedback loop (Algorithm 2) on predictions
produced by the trained neural models on the held-out test fold of CASAS
and SPHERE, and measures per-cycle weighted F1 against the dataset's own
annotated ground-truth labels.

Why this script exists
----------------------
``run_feedback_cycle_ablation.py`` is a controlled synthetic stress-test
of the feedback mechanism with a known injected FP rate. A reviewer can
legitimately ask whether the F1 convergence reported there is a
self-consistency artefact (rules participating in feedback also defining
labels). This script answers that question by:

  * Using gold labels that come exclusively from the datasets
    (CASAS ``activity`` column, SPHERE activity/posture class). No SWRL
    rule participates in label generation.
  * Running the actual HermiT reasoner with all 11 SWRL rules, the
    actual ``run_feedback_cycle`` retraction / threshold-adaptation step,
    and re-serialising the KG between cycles so each reasoner pass sees
    the updated graph.

Usage
-----
::

    PYTHONPATH=. python evaluation/run_feedback_cycle_real.py \\
        --config config/ns_full.yaml \\
        --casas-model-dir outputs/neural_perception/<tag>/casas/activity \\
        --sphere-model-dir outputs/neural_perception/<tag>/sphere \\
        --max-cycles 5 \\
        --outdir outputs/experiments/feedback_cycle_real

At least one of ``--casas-model-dir`` and ``--sphere-model-dir`` is
required.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def _compute_metrics(
    predictions: List[Dict[str, Any]],
    *,
    conf_threshold: float,
    retracted_event_uris: set,
) -> Dict[str, Any]:
    """Weighted F1, correctness, and FP rate over the *active* prediction set.

    A prediction is active iff:
      (a) it has a non-null dataset ground-truth label, and
      (b) its confidence is at or above the (current) feedback threshold, and
      (c) its event URI has not been retracted by an earlier cycle.

    Ground truth is the dataset annotation column. No SWRL rule
    contributes to the label.
    """
    from sklearn.metrics import f1_score

    active: List[Dict[str, Any]] = []
    for p in predictions:
        if p.get("ground_truth_label") is None:
            continue
        if float(p.get("confidence", 0.0)) < conf_threshold:
            continue
        meta = p.get("metadata") or {}
        event_uri = meta.get("event_uri")
        if event_uri and event_uri in retracted_event_uris:
            continue
        active.append(p)

    if not active:
        return {
            "f1_weighted": 0.0,
            "fp_rate": 0.0,
            "correctness": 0.0,
            "active_predictions": 0,
            "total_predictions": len(predictions),
        }

    y_true = [p["ground_truth_label"] for p in active]
    y_pred = [p["predicted_label"] for p in active]

    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    correct = sum(1 for p in active if p["predicted_label"] == p["ground_truth_label"])
    correctness = correct / len(active)
    fp_rate = 1.0 - correctness  # restricted to the active set

    return {
        "f1_weighted": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
        "correctness": round(correctness, 4),
        "active_predictions": len(active),
        "total_predictions": len(predictions),
    }


def _current_threshold(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("reasoning", {}).get("confidence_threshold_feedback", 0.70))


def _serialize_kg_to_disk(cfg: Dict[str, Any], graph: Any, dataset: str) -> None:
    """Re-serialise the in-memory graph so the next reasoner call sees retractions."""
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.rdf_writer import serialize_graph

    kg_cfg = cfg.get("kg", {})
    out_dir = Path(kg_cfg.get("output_dir", "outputs/kg")) / dataset
    out_file = out_dir / "populated_kg.ttl"
    fmt = kg_cfg.get("serialization_format", "turtle")
    serialize_graph(graph, out_file, fmt=fmt)


def run_real_feedback_for_dataset(
    cfg: Dict[str, Any],
    dataset: str,
    task: str,
    model_dir: Path,
    max_cycles: int,
) -> Dict[str, Any]:
    """Run cycles 0..max for one dataset using the real reasoner + feedback."""
    from neurosymbolic_iot.cli.run_pipeline import (
        _run_kg_stage,
        _run_neural_stage,
        _run_reasoning_stage,
    )
    from neurosymbolic_iot.neural_perception.utils import pick_device
    from neurosymbolic_iot.reasoning_feedback.feedback.feedback_loop import (
        RetrainBuffer,
        run_feedback_cycle,
    )

    device = pick_device(str(cfg.get("neural_perception", {}).get("device", "auto")))

    # Stage 1: real neural predictions on the held-out test fold.
    log.info("[%s] running neural inference", dataset)
    predictions = _run_neural_stage(cfg, dataset, task, model_dir, device)
    log.info("[%s] %d predictions produced", dataset, len(predictions))

    # Stage 2: real KG (also serialises to disk).
    log.info("[%s] building knowledge graph", dataset)
    graph = _run_kg_stage(cfg, predictions, dataset)

    # Stage 3: initial real reasoner pass.
    log.info("[%s] running symbolic reasoner (cycle 0 baseline)", dataset)
    reasoning_result = _run_reasoning_stage(cfg, dataset)

    retracted_event_uris: set = set()
    cycle_metrics: List[Dict[str, Any]] = []

    # ---------------- cycle 0 (pre-feedback baseline) ---------------- #
    threshold = _current_threshold(cfg)
    m0 = _compute_metrics(
        predictions,
        conf_threshold=threshold,
        retracted_event_uris=retracted_event_uris,
    )
    m0.update({
        "cycle": 0,
        "conf_threshold": round(threshold, 4),
        "feedback_flags": len(reasoning_result.feedback_flags),
        "contradictions_detected": 0,
        "false_positives_retracted": 0,
    })
    cycle_metrics.append(m0)
    log.info(
        "[%s] cycle=0 F1=%.3f FP=%.3f active=%d threshold=%.3f flags=%d",
        dataset, m0["f1_weighted"], m0["fp_rate"], m0["active_predictions"],
        threshold, len(reasoning_result.feedback_flags),
    )

    buffer = RetrainBuffer(
        max_size=int(cfg.get("feedback", {}).get("retrain_buffer_size", 500)),
        trigger_size=int(cfg.get("feedback", {}).get("retrain_trigger_size", 100)),
    )

    # ---------------- cycles 1..max ---------------- #
    for cycle_id in range(1, max_cycles + 1):
        # Real feedback step: mutates cfg threshold and retracts triples from graph.
        cycle = run_feedback_cycle(
            reasoning_result=reasoning_result,
            predictions=predictions,
            graph=graph,
            buffer=buffer,
            cfg=cfg,
            model_dir=model_dir,
            cycle_id=cycle_id,
        )

        # Track event URIs retracted by FalsePositiveHallucination flags.
        for flag in reasoning_result.feedback_flags:
            if "FalsePositiveHallucination" in flag.get("error_type", ""):
                uri = flag.get("uri")
                if uri:
                    retracted_event_uris.add(str(uri))

        # Re-serialise so the next reasoner pass observes the retractions.
        _serialize_kg_to_disk(cfg, graph, dataset)

        # Re-run the real reasoner over the now-smaller graph.
        log.info("[%s] re-running reasoner after cycle %d", dataset, cycle_id)
        reasoning_result = _run_reasoning_stage(cfg, dataset)

        threshold = _current_threshold(cfg)
        m = _compute_metrics(
            predictions,
            conf_threshold=threshold,
            retracted_event_uris=retracted_event_uris,
        )
        m.update({
            "cycle": cycle_id,
            "conf_threshold": round(threshold, 4),
            "contradictions_detected": cycle.contradictions_detected,
            "false_positives_retracted": cycle.false_positives_retracted,
            "feedback_flags": len(reasoning_result.feedback_flags),
        })
        cycle_metrics.append(m)

        log.info(
            "[%s] cycle=%d F1=%.3f FP=%.3f active=%d threshold=%.3f flags=%d",
            dataset, cycle_id, m["f1_weighted"], m["fp_rate"],
            m["active_predictions"], threshold, len(reasoning_result.feedback_flags),
        )

        if cycle.contradictions_detected == 0 and len(reasoning_result.feedback_flags) == 0:
            log.info("[%s] converged at cycle %d (no remaining contradictions)",
                     dataset, cycle_id)
            break

    return {
        "dataset": dataset,
        "task": task,
        "model_dir": str(model_dir),
        "cycles": cycle_metrics,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-data feedback cycle evaluation (held-out, non-simulated).",
    )
    p.add_argument("--config", type=str, default="config/ns_full.yaml",
                   help="Pipeline config (default: ns_full.yaml).")
    p.add_argument("--casas-model-dir", type=str, default=None,
                   help="Trained CASAS model directory (contains model.pth, vocab.json).")
    p.add_argument("--sphere-model-dir", type=str, default=None,
                   help="Trained SPHERE model directory.")
    p.add_argument("--casas-task", type=str, default="activity",
                   choices=["activity", "transition"])
    p.add_argument("--max-cycles", type=int, default=5)
    p.add_argument("--outdir", type=str,
                   default="outputs/experiments/feedback_cycle_real")
    return p.parse_args()


def main() -> int:
    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging
    from neurosymbolic_iot.utils.seed import set_global_seed

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    log.info("=== Real-Data Feedback Cycle Evaluation ===")
    log.info("Max cycles: %d", args.max_cycles)

    results: Dict[str, Any] = {"datasets": {}, "metadata": {}}
    targets: List[Tuple[str, str, str]] = []
    if args.casas_model_dir:
        targets.append(("casas", args.casas_task, args.casas_model_dir))
    if args.sphere_model_dir:
        targets.append(("sphere", "activity", args.sphere_model_dir))
    if not targets:
        log.error("Provide --casas-model-dir and/or --sphere-model-dir.")
        return 1

    for dataset, task, model_dir in targets:
        log.info("--- Dataset: %s (model=%s) ---", dataset.upper(), model_dir)
        # Fresh cfg per dataset so threshold mutations do not leak across runs.
        cfg_ds = copy.deepcopy(cfg)
        ds_result = run_real_feedback_for_dataset(
            cfg=cfg_ds,
            dataset=dataset,
            task=task,
            model_dir=Path(model_dir),
            max_cycles=args.max_cycles,
        )
        results["datasets"][dataset] = ds_result

    results["metadata"] = {
        "config_path": args.config,
        "max_cycles": args.max_cycles,
        "ground_truth_source": (
            "Dataset annotations only (CASAS activity column, "
            "SPHERE activity/posture class). No SWRL rule contributes "
            "to label generation."
        ),
        "feedback_participating_rules": [8, 9, 10, 11],
        "label_defining_rules": [],
        "rule_label_disjoint": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "feedback_cycle_real_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
