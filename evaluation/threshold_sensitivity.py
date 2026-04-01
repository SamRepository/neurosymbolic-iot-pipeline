from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result from one threshold configuration."""

    threshold: float = 0.0
    validated_events: int = 0
    critical_anomalies: int = 0
    behavioral_anomalies: int = 0
    feedback_flags: int = 0
    fp_rate_pct: float = 0.0
    correctness_pct: float = 0.0


def run_threshold_sensitivity(
    *,
    cfg: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    dataset: str,
    thresholds: List[float] = None,
    out_dir: Path = Path("outputs/experiments"),
) -> List[ThresholdResult]:
    """
    Vary SWRL rule confidence thresholds and measure reasoning outcomes.

    For each threshold value, overrides all three confidence thresholds
    (validation, anomaly, feedback) and re-runs the symbolic reasoner
    + one feedback cycle.

    Parameters
    ----------
    cfg : dict
        Full pipeline config.
    predictions : list of dicts
        Neural predictions (already inferred).
    dataset : str
        "casas" or "sphere".
    thresholds : list of float
        Threshold values to test. Default: [0.60, 0.70, 0.80, 0.85, 0.90, 0.95].
    out_dir : Path
        Where to save results JSON.
    """
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
        build_kg_from_predictions,
        load_sensor_map,
    )
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.rdf_writer import serialize_graph
    from neurosymbolic_iot.reasoning_feedback.feedback.feedback_loop import (
        RetrainBuffer,
        detect_contradictions,
    )
    from neurosymbolic_iot.reasoning_feedback.reasoning.symbolic_reasoner import run_reasoning

    if thresholds is None:
        thresholds = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    kg_cfg = cfg.get("kg", {})
    sensor_map_path = Path(kg_cfg.get("sensor_map_path", "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json"))
    ontology_path = Path(kg_cfg.get("ontology_path", ""))
    kg_out_dir = Path(kg_cfg.get("output_dir", "outputs/kg")) / dataset

    # Build KG once (shared across threshold runs)
    sensor_map = load_sensor_map(sensor_map_path)
    graph = build_kg_from_predictions(
        predictions,
        sensor_map,
        dataset,
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    kg_file = kg_out_dir / "populated_kg_threshold_exp.ttl"
    serialize_graph(graph, kg_file)

    results: List[ThresholdResult] = []

    for thresh in thresholds:
        log.info("Testing threshold=%.2f", thresh)

        # Override all confidence thresholds in a deep copy of the config
        test_cfg = copy.deepcopy(cfg)
        reasoning_section = test_cfg.setdefault("reasoning", {})
        reasoning_section["confidence_threshold_validation"] = thresh
        reasoning_section["confidence_threshold_anomaly"] = thresh
        reasoning_section["confidence_threshold_feedback"] = 1.0 - thresh  # inverse for feedback detection

        # Run reasoning
        rr = run_reasoning(ontology_path, kg_file, cfg=test_cfg)

        # Detect contradictions
        contradictions = detect_contradictions(rr, predictions)

        total_classified = (
            len(rr.validated_events)
            + len(rr.critical_anomalies)
            + len(rr.behavioral_anomalies)
            + len(rr.feedback_flags)
        )
        correctness = (len(rr.validated_events) / max(total_classified, 1)) * 100.0
        fp_rate = (len(contradictions) / max(len(predictions), 1)) * 100.0

        result = ThresholdResult(
            threshold=thresh,
            validated_events=len(rr.validated_events),
            critical_anomalies=len(rr.critical_anomalies),
            behavioral_anomalies=len(rr.behavioral_anomalies),
            feedback_flags=len(rr.feedback_flags),
            fp_rate_pct=round(fp_rate, 2),
            correctness_pct=round(correctness, 2),
        )
        results.append(result)
        log.info(
            "  threshold=%.2f: validated=%d, critical=%d, feedback=%d, FP=%.1f%%, correctness=%.1f%%",
            thresh, result.validated_events, result.critical_anomalies,
            result.feedback_flags, result.fp_rate_pct, result.correctness_pct,
        )

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "threshold_sensitivity.json"
    out_file.write_text(
        json.dumps(
            {"dataset": dataset, "thresholds_tested": thresholds, "results": [asdict(r) for r in results]},
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Threshold sensitivity results saved to %s", out_file)

    return results
