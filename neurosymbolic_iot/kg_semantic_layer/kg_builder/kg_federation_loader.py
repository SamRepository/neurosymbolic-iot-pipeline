from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
TIME = Namespace("http://www.w3.org/2006/time#")


def load_sensor_map(path: Path) -> Dict[str, Any]:
    """Load sensor_map.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_sensor_type(sensor_id: str, patterns: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Match a raw sensor ID against regex patterns from sensor_map."""
    for pattern, info in patterns.items():
        if re.fullmatch(pattern, sensor_id):
            return info
    return None


def _is_night_time(ts_str: Optional[str]) -> bool:
    """Check if a timestamp string falls between 22:00 and 06:00."""
    if not ts_str:
        return False
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.hour >= 22 or dt.hour < 6
    except (ValueError, TypeError):
        return False


def build_kg_from_predictions(
    predictions: List[Dict[str, Any]],
    sensor_map: Dict[str, Any],
    dataset: str,
    *,
    ontology_path: Optional[Path] = None,
    person_uri: Optional[str] = None,
) -> Graph:
    """
    Build an rdflib Graph from neural predictions (Algorithm 1 from paper).

    Parameters
    ----------
    predictions : list of dicts with keys: predicted_label, confidence,
        probabilities, window_start, window_end, metadata
    sensor_map : parsed sensor_map.json
    dataset : "casas" or "sphere"
    ontology_path : if given, parse the TTL ontology into the graph first
    person_uri : URI for the person individual (default: nsiot:Participant1)

    Returns
    -------
    rdflib.Graph populated with NeuralPrediction and Event individuals
    """
    g = Graph()
    g.bind("nsiot", NSIOT)
    g.bind("sosa", SOSA)
    g.bind("time", TIME)
    g.bind("owl", OWL)

    if ontology_path and Path(ontology_path).exists():
        g.parse(str(ontology_path), format="turtle")
        log.info("Loaded ontology from %s (%d triples)", ontology_path, len(g))

    person_ref = URIRef(person_uri) if person_uri else NSIOT["Participant1"]
    g.add((person_ref, RDF.type, NSIOT["Person"]))

    ds_map = sensor_map.get(dataset, {})
    activity_map = ds_map.get("activity_map", {})

    for idx, pred in enumerate(predictions):
        _add_prediction_to_graph(g, pred, ds_map, activity_map, dataset, person_ref, idx)

    log.info("Built KG with %d triples from %d predictions (dataset=%s)", len(g), len(predictions), dataset)
    return g


def _add_prediction_to_graph(
    g: Graph,
    pred: Dict[str, Any],
    ds_map: Dict[str, Any],
    activity_map: Dict[str, str],
    dataset: str,
    person_ref: URIRef,
    idx: int,
) -> None:
    """Add one NeuralPrediction + one Event to the graph."""
    pred_uri = NSIOT[f"pred_{idx}"]
    event_uri = NSIOT[f"event_{idx}"]

    def _resolve_activity_iri(raw: str) -> URIRef:
        onto_label = activity_map.get(raw, raw)
        if isinstance(onto_label, str) and onto_label.startswith("nsiot:"):
            return NSIOT[onto_label.split(":", 1)[1]]
        return NSIOT[onto_label]

    # --- Primary NeuralPrediction individual (top-1) ---
    g.add((pred_uri, RDF.type, NSIOT["NeuralPrediction"]))
    g.add((pred_uri, NSIOT["isAlternativePrediction"], Literal(False, datatype=XSD.boolean)))

    raw_label = pred.get("predicted_label", "")
    g.add((pred_uri, NSIOT["predictsActivity"], _resolve_activity_iri(raw_label)))

    confidence = pred.get("confidence", 0.0)
    g.add((pred_uri, NSIOT["hasConfidenceScore"], Literal(float(confidence), datatype=XSD.float)))

    model_tag = pred.get("metadata", {}).get("model_tag", "unknown")
    g.add((pred_uri, NSIOT["generatedByModel"], Literal(str(model_tag), datatype=XSD.string)))

    window_start = pred.get("window_start")
    if window_start:
        g.add((pred_uri, NSIOT["generatedAt"], Literal(window_start, datatype=XSD.dateTime)))

    # --- Alternative (top-2) prediction so rule 10 (mutually-exclusive
    # top-2 with low margin) is evaluable. ``probabilities`` is the full
    # softmax over id2label; we recover the second-highest class. We need
    # id2label in the same order — passed via the prediction dict under
    # metadata.id2label when emitted by the inference path, with the raw
    # ``probabilities`` field already populated.
    probs = pred.get("probabilities") or []
    id2label_meta = (pred.get("metadata", {}) or {}).get("id2label")
    if probs and id2label_meta and len(probs) == len(id2label_meta):
        top1 = max(range(len(probs)), key=lambda i: probs[i])
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        for j in ranked:
            if j != top1:
                top2 = j
                break
        else:
            top2 = top1
        alt_uri = NSIOT[f"pred_{idx}_alt"]
        g.add((alt_uri, RDF.type, NSIOT["NeuralPrediction"]))
        g.add((alt_uri, NSIOT["isAlternativePrediction"], Literal(True, datatype=XSD.boolean)))
        g.add((alt_uri, NSIOT["predictsActivity"],
               _resolve_activity_iri(str(id2label_meta[top2]))))
        g.add((alt_uri, NSIOT["hasConfidenceScore"],
               Literal(float(probs[top2]), datatype=XSD.float)))

    # --- Event individual ---
    g.add((event_uri, RDF.type, NSIOT["Event"]))
    g.add((event_uri, NSIOT["involvesPerson"], person_ref))
    g.add((event_uri, NSIOT["isBasedOnPrediction"], pred_uri))
    if probs and id2label_meta and len(probs) == len(id2label_meta):
        g.add((event_uri, NSIOT["isBasedOnPrediction"], NSIOT[f"pred_{idx}_alt"]))

    # Temporal entity
    if window_start:
        time_uri = NSIOT[f"time_{idx}"]
        g.add((time_uri, RDF.type, TIME["Instant"]))
        g.add((time_uri, TIME["inXSDDateTimeStamp"], Literal(window_start, datatype=XSD.dateTime)))
        g.add((event_uri, NSIOT["hasTemporalEntity"], time_uri))

    # Night time context
    if _is_night_time(window_start):
        ctx_uri = NSIOT[f"nightctx_{idx}"]
        g.add((ctx_uri, RDF.type, NSIOT["NightTimeContext"]))
        g.add((event_uri, NSIOT["hasTimeContext"], ctx_uri))

    # --- Sensor observations from metadata ---
    metadata = pred.get("metadata", {})
    if dataset == "casas":
        _add_casas_sensors(g, metadata, ds_map, event_uri)
    elif dataset == "sphere":
        _add_sphere_sensors(g, metadata, ds_map, event_uri)


def _add_casas_sensors(
    g: Graph,
    metadata: Dict[str, Any],
    ds_map: Dict[str, Any],
    event_uri: URIRef,
) -> None:
    """Add CASAS sensor observations from window metadata."""
    sensor_tokens = metadata.get("sensor_tokens", [])
    patterns = ds_map.get("sensor_type_patterns", {})
    room_map = ds_map.get("room_assignments", {})

    seen_sensors: set = set()
    for token in sensor_tokens:
        parts = str(token).split(":", 1) if ":" in str(token) else str(token).split("=", 1)
        if len(parts) != 2:
            continue
        sensor_id, state_val = parts[0].strip(), parts[1].strip()
        if sensor_id in seen_sensors:
            continue
        seen_sensors.add(sensor_id)

        sensor_uri = NSIOT[f"sensor_{sensor_id}"]
        type_info = _resolve_sensor_type(sensor_id, patterns)
        if type_info:
            rdf_type = type_info["rdf_type"]
            if rdf_type.startswith("nsiot:"):
                g.add((sensor_uri, RDF.type, NSIOT[rdf_type.split(":", 1)[1]]))
            state_map = type_info.get("state_map", {})
            mapped_state = state_map.get(state_val.upper())
            if mapped_state and mapped_state.startswith("nsiot:"):
                g.add((sensor_uri, NSIOT["hasState"], NSIOT[mapped_state.split(":", 1)[1]]))

        if sensor_id in room_map:
            room_str = room_map[sensor_id]
            if room_str.startswith("nsiot:"):
                g.add((sensor_uri, NSIOT["isLocatedIn"], NSIOT[room_str.split(":", 1)[1]]))


def _add_sphere_sensors(
    g: Graph,
    metadata: Dict[str, Any],
    ds_map: Dict[str, Any],
    event_uri: URIRef,
) -> None:
    """Add SPHERE PIR sensor observations from window metadata."""
    pir_columns = ds_map.get("pir_columns", {})
    active_pirs = metadata.get("active_pirs", [])

    for pir_name in active_pirs:
        pir_info = pir_columns.get(pir_name)
        if not pir_info:
            continue
        sensor_uri = NSIOT[f"pir_{pir_name}"]
        g.add((sensor_uri, RDF.type, NSIOT["PIRMotionSensor"]))
        g.add((sensor_uri, NSIOT["hasState"], NSIOT["StateMotionDetected"]))
        room_str = pir_info.get("room", "")
        if room_str.startswith("nsiot:"):
            g.add((sensor_uri, NSIOT["isLocatedIn"], NSIOT[room_str.split(":", 1)[1]]))
