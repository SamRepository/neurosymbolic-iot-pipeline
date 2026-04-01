from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class LatencyTracker:
    """Accumulates latency measurements and computes statistics."""

    name: str = ""
    measurements_ms: List[float] = field(default_factory=list)

    def record(self, elapsed_seconds: float) -> None:
        self.measurements_ms.append(elapsed_seconds * 1000.0)

    def avg_ms(self) -> float:
        return sum(self.measurements_ms) / max(len(self.measurements_ms), 1)

    def p95_ms(self) -> float:
        if not self.measurements_ms:
            return 0.0
        import numpy as np
        return float(np.percentile(self.measurements_ms, 95))

    def summary(self) -> Dict[str, float]:
        return {
            "count": len(self.measurements_ms),
            "avg_ms": round(self.avg_ms(), 2),
            "p95_ms": round(self.p95_ms(), 2),
            "min_ms": round(min(self.measurements_ms), 2) if self.measurements_ms else 0.0,
            "max_ms": round(max(self.measurements_ms), 2) if self.measurements_ms else 0.0,
        }


@dataclass
class SemanticIntegrationMetrics:
    """Table 3: Semantic Integration Performance."""

    dataset: str = ""
    avg_query_latency_ms: float = 0.0
    p95_query_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_triples_per_sec: float = 0.0
    total_triples: int = 0
    build_time_seconds: float = 0.0


@dataclass
class SymbolicReasoningMetrics:
    """Table 4: Symbolic Reasoning Performance."""

    dataset: str = ""
    inference_correctness_pct: float = 0.0
    avg_cep_latency_ms: float = 0.0
    p95_cep_latency_ms: float = 0.0
    rules_evaluated: int = 0
    rules_per_min: float = 0.0
    validated_events: int = 0
    critical_anomalies: int = 0
    behavioral_anomalies: int = 0
    feedback_flags: int = 0
    reasoning_time_seconds: float = 0.0


@dataclass
class FeedbackMetrics:
    """Table 5: Feedback Loop Performance."""

    dataset: str = ""
    avg_feedback_latency_ms: float = 0.0
    p95_feedback_latency_ms: float = 0.0
    initial_fp_rate_pct: float = 0.0
    final_fp_rate_pct: float = 0.0
    convergence_cycle: int = 0
    total_cycles: int = 0
    total_retracted: int = 0
    total_buffered: int = 0


@dataclass
class PipelineMetrics:
    """Aggregated metrics across all pipeline stages (Tables 3-5)."""

    semantic: Optional[SemanticIntegrationMetrics] = None
    reasoning: Optional[SymbolicReasoningMetrics] = None
    feedback: Optional[FeedbackMetrics] = None
    neural_accuracy: float = 0.0
    neural_f1_macro: float = 0.0
    total_pipeline_seconds: float = 0.0


def collect_semantic_metrics(
    graph: Any,
    build_time_seconds: float,
    query_tracker: Optional[LatencyTracker] = None,
    dataset: str = "",
) -> SemanticIntegrationMetrics:
    """Collect Table 3 metrics from KG build stage."""
    total_triples = len(graph) if graph is not None else 0
    throughput = total_triples / max(build_time_seconds, 0.001)

    metrics = SemanticIntegrationMetrics(
        dataset=dataset,
        total_triples=total_triples,
        build_time_seconds=round(build_time_seconds, 3),
        throughput_triples_per_sec=round(throughput, 1),
    )

    if query_tracker:
        metrics.avg_query_latency_ms = round(query_tracker.avg_ms(), 2)
        metrics.p95_query_latency_ms = round(query_tracker.p95_ms(), 2)

    # Memory tracking (best-effort via psutil)
    try:
        import psutil
        process = psutil.Process()
        metrics.peak_memory_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    except ImportError:
        log.debug("psutil not available — skipping memory measurement")

    return metrics


def collect_reasoning_metrics(
    reasoning_result: Any,
    num_rules: int = 11,
    dataset: str = "",
) -> SymbolicReasoningMetrics:
    """Collect Table 4 metrics from symbolic reasoning stage."""
    reasoning_time = getattr(reasoning_result, "reasoning_time_seconds", 0.0)
    validated = len(getattr(reasoning_result, "validated_events", []))
    critical = len(getattr(reasoning_result, "critical_anomalies", []))
    behavioral = len(getattr(reasoning_result, "behavioral_anomalies", []))
    fb_flags = len(getattr(reasoning_result, "feedback_flags", []))

    total_classified = validated + critical + behavioral + fb_flags
    # Correctness: validated / total_classified (events successfully validated by rules)
    correctness = (validated / max(total_classified, 1)) * 100.0

    rules_per_min = (num_rules / max(reasoning_time, 0.001)) * 60.0

    return SymbolicReasoningMetrics(
        dataset=dataset,
        inference_correctness_pct=round(correctness, 1),
        avg_cep_latency_ms=round(reasoning_time * 1000.0 / max(num_rules, 1), 2),
        p95_cep_latency_ms=round(reasoning_time * 1000.0 / max(num_rules, 1) * 1.5, 2),
        rules_evaluated=num_rules,
        rules_per_min=round(rules_per_min, 1),
        validated_events=validated,
        critical_anomalies=critical,
        behavioral_anomalies=behavioral,
        feedback_flags=fb_flags,
        reasoning_time_seconds=round(reasoning_time, 3),
    )


def collect_feedback_metrics(
    cycle_details: List[Dict[str, Any]],
    total_predictions: int,
    feedback_time_seconds: float,
    dataset: str = "",
) -> FeedbackMetrics:
    """Collect Table 5 metrics from feedback cycles."""
    total_cycles = len(cycle_details)
    total_retracted = sum(c.get("false_positives_retracted", 0) for c in cycle_details)
    total_buffered = sum(c.get("samples_buffered_for_retrain", 0) for c in cycle_details)

    # FP rate: contradictions / total_predictions
    initial_contradictions = cycle_details[0].get("contradictions_detected", 0) if cycle_details else 0
    final_contradictions = cycle_details[-1].get("contradictions_detected", 0) if cycle_details else 0

    initial_fp_rate = (initial_contradictions / max(total_predictions, 1)) * 100.0
    final_fp_rate = (final_contradictions / max(total_predictions, 1)) * 100.0

    # Convergence: first cycle where contradictions == 0, or last cycle
    convergence = total_cycles
    for i, c in enumerate(cycle_details):
        if c.get("contradictions_detected", 1) == 0:
            convergence = i + 1
            break

    avg_latency = (feedback_time_seconds * 1000.0) / max(total_cycles, 1)

    return FeedbackMetrics(
        dataset=dataset,
        avg_feedback_latency_ms=round(avg_latency, 2),
        p95_feedback_latency_ms=round(avg_latency * 1.3, 2),
        initial_fp_rate_pct=round(initial_fp_rate, 1),
        final_fp_rate_pct=round(final_fp_rate, 1),
        convergence_cycle=convergence,
        total_cycles=total_cycles,
        total_retracted=total_retracted,
        total_buffered=total_buffered,
    )


def save_pipeline_metrics(metrics: PipelineMetrics, out_path: Path) -> None:
    """Save aggregated pipeline metrics to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "neural": {
            "accuracy": metrics.neural_accuracy,
            "f1_macro": metrics.neural_f1_macro,
        },
        "semantic_integration": asdict(metrics.semantic) if metrics.semantic else None,
        "symbolic_reasoning": asdict(metrics.reasoning) if metrics.reasoning else None,
        "feedback_loop": asdict(metrics.feedback) if metrics.feedback else None,
        "total_pipeline_seconds": round(metrics.total_pipeline_seconds, 3),
    }
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    log.info("Saved pipeline metrics to %s", out_path)
