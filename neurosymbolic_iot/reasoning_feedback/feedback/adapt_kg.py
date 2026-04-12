from __future__ import annotations

import logging
from typing import Any, Dict, List

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")


def retract_false_positives(
    g: Graph,
    feedback_flags: List[Dict[str, Any]],
) -> int:
    """
    Remove triples associated with FeedbackRequired events flagged as
    FalsePositiveHallucination. Returns number of triples removed.
    """
    removed = 0
    for flag in feedback_flags:
        error_type = flag.get("error_type", "")
        if "FalsePositiveHallucination" not in error_type:
            continue
        event_uri = URIRef(flag["uri"])
        # Remove all triples where this event is a subject
        triples_to_remove = list(g.triples((event_uri, None, None)))
        for t in triples_to_remove:
            g.remove(t)
            removed += 1
        # Also remove the associated prediction
        for _, _, pred in g.triples((event_uri, NSIOT["isBasedOnPrediction"], None)):
            pred_triples = list(g.triples((pred, None, None)))
            for t in pred_triples:
                g.remove(t)
                removed += 1

    log.info("Retracted %d triples from %d false positive events", removed, len(feedback_flags))
    return removed


def demote_low_confidence_predictions(
    g: Graph,
    threshold: float = 0.5,
) -> int:
    """
    Find NeuralPrediction instances with confidence below threshold
    and remove their predictsActivity/predictsPosture assertions.
    """
    demoted = 0
    for pred_uri in g.subjects(RDF.type, NSIOT["NeuralPrediction"]):
        for conf_lit in g.objects(pred_uri, NSIOT["hasConfidenceScore"]):
            conf = float(conf_lit)
            if conf < threshold:
                for t in list(g.triples((pred_uri, NSIOT["predictsActivity"], None))):
                    g.remove(t)
                    demoted += 1
                for t in list(g.triples((pred_uri, NSIOT["predictsPosture"], None))):
                    g.remove(t)
                    demoted += 1
    log.info("Demoted %d low-confidence prediction assertions (threshold=%.2f)", demoted, threshold)
    return demoted


def update_confidence_threshold(
    cfg: Dict[str, Any],
    feedback_flags: List[Dict[str, Any]],
    *,
    adjustment_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Adjust confidence thresholds in config based on feedback.
    Too many false positives -> raise threshold.
    """
    reasoning_cfg = cfg.get("reasoning", {})
    fp_count = sum(
        1 for f in feedback_flags
        if "FalsePositiveHallucination" in f.get("error_type", "")
    )
    total = max(len(feedback_flags), 1)
    fp_ratio = fp_count / total

    if fp_ratio > 0.3:
        delta = adjustment_rate
        log.info("High FP ratio (%.2f) — raising thresholds by %.2f", fp_ratio, delta)
    elif fp_ratio < 0.1:
        delta = -adjustment_rate * 0.5
        log.info("Low FP ratio (%.2f) — lowering thresholds by %.2f", fp_ratio, abs(delta))
    else:
        delta = 0.0

    for key in ("confidence_threshold_validation", "confidence_threshold_anomaly", "confidence_threshold_feedback"):
        old = float(reasoning_cfg.get(key, 0.85))
        reasoning_cfg[key] = round(min(0.99, max(0.5, old + delta)), 3)

    cfg["reasoning"] = reasoning_cfg
    return cfg
