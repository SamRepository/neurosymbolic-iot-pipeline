from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class RetrainBuffer:
    """Accumulates misclassified samples for retraining."""

    samples: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 500
    trigger_size: int = 100

    def add(self, sample: Dict[str, Any]) -> None:
        if len(self.samples) < self.max_size:
            self.samples.append(sample)

    def should_trigger_retrain(self) -> bool:
        return len(self.samples) >= self.trigger_size

    def flush(self) -> List[Dict[str, Any]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2, default=str)

    def load(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.samples = json.load(f)


@dataclass
class FeedbackCycle:
    """Results from one feedback loop iteration."""

    cycle_id: int = 0
    contradictions_detected: int = 0
    false_positives_retracted: int = 0
    samples_buffered_for_retrain: int = 0
    retrain_triggered: bool = False
    updated_thresholds: Dict[str, float] = field(default_factory=dict)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Optional[Dict[str, float]] = None


def detect_contradictions(
    reasoning_result: Any,
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compare reasoner output against neural predictions to find contradictions:
    1. OWL inconsistencies (from reasoner)
    2. FeedbackRequired flags from SWRL rules 8-11
    3. Confidence-below-threshold predictions that couldn't be validated
    """
    contradictions: List[Dict[str, Any]] = []

    # OWL inconsistencies
    for msg in reasoning_result.inconsistencies:
        contradictions.append({
            "type": "owl_inconsistency",
            "message": msg,
            "severity": "critical",
        })

    # SWRL-derived feedback flags
    for flag in reasoning_result.feedback_flags:
        error_type = flag.get("error_type", "unknown")
        contradictions.append({
            "type": "swrl_feedback",
            "event_uri": flag.get("uri"),
            "error_type": error_type,
            "confidence": flag.get("confidence"),
            "severity": "high" if "Hallucination" in error_type else "medium",
        })

    # Unvalidated high-confidence predictions (neural says yes, symbolic can't confirm)
    validated_uris = {v.get("uri") for v in reasoning_result.validated_events}
    for pred in predictions:
        meta = pred.get("metadata", {})
        event_uri = meta.get("event_uri")
        if event_uri and event_uri not in validated_uris and pred.get("confidence", 0) > 0.7:
            contradictions.append({
                "type": "unvalidated_prediction",
                "event_uri": event_uri,
                "predicted_label": pred.get("predicted_label"),
                "confidence": pred.get("confidence"),
                "severity": "low",
            })

    log.info("Detected %d contradictions", len(contradictions))
    return contradictions


def buffer_misclassified_samples(
    contradictions: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    buffer: RetrainBuffer,
) -> int:
    """Add misclassified samples to the retrain buffer."""
    added = 0
    contradiction_uris = {
        c.get("event_uri")
        for c in contradictions
        if c.get("type") in ("swrl_feedback", "unvalidated_prediction")
    }
    for pred in predictions:
        event_uri = pred.get("metadata", {}).get("event_uri")
        if event_uri and event_uri in contradiction_uris:
            buffer.add({
                "sample_idx": pred.get("sample_idx"),
                "predicted_label": pred.get("predicted_label"),
                "ground_truth_label": pred.get("ground_truth_label"),
                "confidence": pred.get("confidence"),
                "window_start": pred.get("window_start"),
                "window_end": pred.get("window_end"),
            })
            added += 1
    log.info("Buffered %d samples for retraining (buffer size: %d)", added, len(buffer.samples))
    return added


def trigger_retrain(
    buffer: RetrainBuffer,
    cfg: Dict[str, Any],
    model_dir: Path,
) -> Optional[Dict[str, float]]:
    """
    If buffer exceeds threshold, trigger retraining.
    Returns new test metrics if retrain happened, None otherwise.
    """
    if not buffer.should_trigger_retrain():
        log.info("Buffer size %d < trigger %d — skipping retrain", len(buffer.samples), buffer.trigger_size)
        return None

    log.info("Retrain triggered with %d buffered samples", len(buffer.samples))
    samples = buffer.flush()

    # Save buffer for offline retraining
    buffer_path = Path(model_dir) / "retrain_buffer.json"
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(buffer_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, default=str)
    log.info("Saved retrain buffer to %s", buffer_path)

    # Note: actual retraining via train_loop() would be called here.
    # For now, we log the trigger and return a placeholder.
    return {"retrain_samples": len(samples), "status": "buffer_saved"}


def run_feedback_cycle(
    *,
    reasoning_result: Any,
    predictions: List[Dict[str, Any]],
    graph: Any,
    buffer: RetrainBuffer,
    cfg: Dict[str, Any],
    model_dir: Path,
    cycle_id: int = 0,
) -> FeedbackCycle:
    """Execute one full feedback cycle (Algorithm 2 from paper, lines 6-13)."""
    from neurosymbolic_iot.reasoning_feedback.feedback.adapt_kg import (
        retract_false_positives,
        update_confidence_threshold,
    )

    cycle = FeedbackCycle(cycle_id=cycle_id)

    # 1. Detect contradictions
    contradictions = detect_contradictions(reasoning_result, predictions)
    cycle.contradictions_detected = len(contradictions)

    # 2. Retract false positives from KG
    if graph is not None:
        cycle.false_positives_retracted = retract_false_positives(
            graph, reasoning_result.feedback_flags
        )

    # 3. Buffer misclassified samples
    cycle.samples_buffered_for_retrain = buffer_misclassified_samples(
        contradictions, predictions, buffer
    )

    # 4. Optionally trigger retrain
    retrain_result = trigger_retrain(buffer, cfg, model_dir)
    cycle.retrain_triggered = retrain_result is not None
    cycle.metrics_after = retrain_result

    # 5. Update confidence thresholds
    cfg = update_confidence_threshold(
        cfg,
        reasoning_result.feedback_flags,
        adjustment_rate=float(cfg.get("feedback", {}).get("confidence_adjustment_rate", 0.05)),
    )
    cycle.updated_thresholds = {
        k: v
        for k, v in cfg.get("reasoning", {}).items()
        if "threshold" in k
    }

    log.info(
        "Feedback cycle %d: %d contradictions, %d retracted, %d buffered, retrain=%s",
        cycle_id, cycle.contradictions_detected, cycle.false_positives_retracted,
        cycle.samples_buffered_for_retrain, cycle.retrain_triggered,
    )
    return cycle
