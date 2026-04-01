from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")


@dataclass
class FederatedResult:
    """Comparison of single-source vs. federated reasoning."""

    casas_only: Dict[str, Any] = field(default_factory=dict)
    sphere_only: Dict[str, Any] = field(default_factory=dict)
    federated: Dict[str, Any] = field(default_factory=dict)
    federation_gain: Dict[str, float] = field(default_factory=dict)


def _count_typed_individuals(g: Graph, class_name: str) -> int:
    """Count individuals of a given nsiot class in the graph."""
    cls_uri = NSIOT[class_name]
    return sum(1 for _ in g.triples((None, RDF.type, cls_uri)))


def _graph_summary(g: Graph) -> Dict[str, Any]:
    """Quick summary of reasoning-relevant individuals in a graph."""
    return {
        "total_triples": len(g),
        "neural_predictions": _count_typed_individuals(g, "NeuralPrediction"),
        "events": _count_typed_individuals(g, "Event"),
        "validated_events": _count_typed_individuals(g, "ValidatedEvent"),
        "critical_anomalies": _count_typed_individuals(g, "CriticalAnomaly"),
        "behavioral_anomalies": _count_typed_individuals(g, "BehavioralAnomaly"),
        "feedback_required": _count_typed_individuals(g, "FeedbackRequired"),
        "persons": _count_typed_individuals(g, "Person"),
    }


def build_federated_graph(
    casas_predictions: List[Dict[str, Any]],
    sphere_predictions: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Graph:
    """
    Build a single federated RDF graph from both CASAS and SPHERE predictions.

    This implements the paper's "federated knowledge integration" claim
    by merging triples from both datasets into a shared ontology namespace.
    """
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
        build_kg_from_predictions,
        load_sensor_map,
    )

    kg_cfg = cfg.get("kg", {})
    sensor_map_path = Path(kg_cfg.get("sensor_map_path", "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json"))
    ontology_path = Path(kg_cfg.get("ontology_path", ""))

    sensor_map = load_sensor_map(sensor_map_path)

    # Build individual dataset graphs
    g_casas = build_kg_from_predictions(
        casas_predictions,
        sensor_map,
        "casas",
        ontology_path=ontology_path if ontology_path.exists() else None,
        person_uri="http://example.org/neuro-symbolic-iot#CASASParticipant1",
    )

    # Build SPHERE graph without re-loading ontology (already in casas graph)
    g_sphere = build_kg_from_predictions(
        sphere_predictions,
        sensor_map,
        "sphere",
        person_uri="http://example.org/neuro-symbolic-iot#SPHEREParticipant1",
    )

    # Merge: add all SPHERE triples into the CASAS graph
    federated = Graph()
    federated.bind("nsiot", NSIOT)

    # Load ontology first
    if ontology_path.exists():
        federated.parse(str(ontology_path), format="turtle")

    for triple in g_casas:
        federated.add(triple)
    for triple in g_sphere:
        federated.add(triple)

    log.info(
        "Federated graph: %d triples (CASAS=%d, SPHERE=%d, ontology+merged=%d)",
        len(federated), len(g_casas), len(g_sphere), len(federated),
    )
    return federated


def run_federated_experiment(
    *,
    casas_predictions: List[Dict[str, Any]],
    sphere_predictions: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_dir: Path = Path("outputs/experiments"),
) -> FederatedResult:
    """
    Compare single-source vs. federated anomaly detection rates.

    Steps:
    1. Build CASAS-only KG -> reason -> count anomalies
    2. Build SPHERE-only KG -> reason -> count anomalies
    3. Build federated KG (both) -> reason -> count anomalies
    4. Compare anomaly detection improvement from federation
    """
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
        build_kg_from_predictions,
        load_sensor_map,
    )
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.rdf_writer import serialize_graph
    from neurosymbolic_iot.reasoning_feedback.reasoning.symbolic_reasoner import run_reasoning

    kg_cfg = cfg.get("kg", {})
    sensor_map_path = Path(kg_cfg.get("sensor_map_path", "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json"))
    ontology_path = Path(kg_cfg.get("ontology_path", ""))
    sensor_map = load_sensor_map(sensor_map_path)

    exp_dir = out_dir / "federated_reasoning"
    exp_dir.mkdir(parents=True, exist_ok=True)

    result = FederatedResult()

    # --- CASAS only ---
    log.info("=== Federated Experiment: CASAS-only reasoning ===")
    g_casas = build_kg_from_predictions(
        casas_predictions, sensor_map, "casas",
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    casas_kg = exp_dir / "casas_only.ttl"
    serialize_graph(g_casas, casas_kg)
    rr_casas = run_reasoning(ontology_path, casas_kg, cfg=cfg)
    result.casas_only = {
        "triples": len(g_casas),
        "validated": len(rr_casas.validated_events),
        "critical_anomalies": len(rr_casas.critical_anomalies),
        "behavioral_anomalies": len(rr_casas.behavioral_anomalies),
        "feedback_flags": len(rr_casas.feedback_flags),
        "inferred_triples": rr_casas.num_inferred_triples,
        "reasoning_time_s": round(rr_casas.reasoning_time_seconds, 3),
    }

    # --- SPHERE only ---
    log.info("=== Federated Experiment: SPHERE-only reasoning ===")
    g_sphere = build_kg_from_predictions(
        sphere_predictions, sensor_map, "sphere",
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    sphere_kg = exp_dir / "sphere_only.ttl"
    serialize_graph(g_sphere, sphere_kg)
    rr_sphere = run_reasoning(ontology_path, sphere_kg, cfg=cfg)
    result.sphere_only = {
        "triples": len(g_sphere),
        "validated": len(rr_sphere.validated_events),
        "critical_anomalies": len(rr_sphere.critical_anomalies),
        "behavioral_anomalies": len(rr_sphere.behavioral_anomalies),
        "feedback_flags": len(rr_sphere.feedback_flags),
        "inferred_triples": rr_sphere.num_inferred_triples,
        "reasoning_time_s": round(rr_sphere.reasoning_time_seconds, 3),
    }

    # --- Federated ---
    log.info("=== Federated Experiment: Combined reasoning ===")
    g_fed = build_federated_graph(casas_predictions, sphere_predictions, cfg)
    fed_kg = exp_dir / "federated.ttl"
    serialize_graph(g_fed, fed_kg)
    rr_fed = run_reasoning(ontology_path, fed_kg, cfg=cfg)
    result.federated = {
        "triples": len(g_fed),
        "validated": len(rr_fed.validated_events),
        "critical_anomalies": len(rr_fed.critical_anomalies),
        "behavioral_anomalies": len(rr_fed.behavioral_anomalies),
        "feedback_flags": len(rr_fed.feedback_flags),
        "inferred_triples": rr_fed.num_inferred_triples,
        "reasoning_time_s": round(rr_fed.reasoning_time_seconds, 3),
    }

    # --- Compute federation gain ---
    sum_single_anomalies = (
        len(rr_casas.critical_anomalies) + len(rr_casas.behavioral_anomalies)
        + len(rr_sphere.critical_anomalies) + len(rr_sphere.behavioral_anomalies)
    )
    fed_anomalies = len(rr_fed.critical_anomalies) + len(rr_fed.behavioral_anomalies)

    sum_single_validated = len(rr_casas.validated_events) + len(rr_sphere.validated_events)
    fed_validated = len(rr_fed.validated_events)

    result.federation_gain = {
        "anomaly_detection_delta": fed_anomalies - sum_single_anomalies,
        "validated_events_delta": fed_validated - sum_single_validated,
        "additional_inferred_triples": (
            rr_fed.num_inferred_triples
            - rr_casas.num_inferred_triples
            - rr_sphere.num_inferred_triples
        ),
    }

    # Save
    out_file = out_dir / "federated_reasoning.json"
    out_file.write_text(
        json.dumps(asdict(result), indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Federated reasoning results saved to %s", out_file)
    log.info(
        "Federation gain: anomalies=%+d, validated=%+d, inferred=%+d",
        result.federation_gain["anomaly_detection_delta"],
        result.federation_gain["validated_events_delta"],
        result.federation_gain["additional_inferred_triples"],
    )

    return result
