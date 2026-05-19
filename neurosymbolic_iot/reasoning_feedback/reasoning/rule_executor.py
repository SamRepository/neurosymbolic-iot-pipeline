"""
Python SWRL-rule executor.

owlready2's bundled HermiT and Pellet only run OWL-DL classification
through ``sync_reasoner_*``; they do not execute SWRL rule bodies. This
module fills that gap by translating each of the 11 SWRL rules from
``symbolic_reasoner.define_swrl_rules`` into a SPARQL ``CONSTRUCT``
query over the populated rdflib graph and firing them in a fixed-point
loop until no new triples are produced.

The output mirrors :class:`ReasoningResult` so the rest of the
pipeline (feedback loop, metric extraction) is unchanged.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from neurosymbolic_iot.reasoning_feedback.reasoning.symbolic_reasoner import (
    ReasoningResult,
)

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")

PREFIXES = """
PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX time:  <http://www.w3.org/2006/time#>
"""


# ---------------------------------------------------------------------------
# Rule definitions: (rule_id, label, swrl-equivalent body+head as SPARQL).
#
# Each entry returns triples to add to the graph. Fired in a fixed-point
# loop. swrlb numeric comparisons (>, <) are expressed via SPARQL FILTER.
# ---------------------------------------------------------------------------

# --- CASAS-Kyoto-ADL-native instantiation of the 11-rule programme. -------
# Rules 1, 2 (sensor grounding) carry over unchanged. Rules 3-11 are
# rewritten to operate on the actual class vocabulary the CASAS GRU
# emits ({MealPreparation, Eating, Housekeeping, PhoneCall} via the
# activity_map) and the actual sensor/state/room individuals the KG
# builder asserts. Posture references (Lying, Standing) and the
# `Sleeping` class — none of which appear in CASAS Kyoto ADL — are
# replaced by domain-justified room/state/confidence patterns. Each
# rule retains its original category (Sensor Grounding / Validation /
# AAL Anomaly / Feedback Trigger) and its original AAL motivation.
RULES: List[Tuple[int, str, str]] = [
    # Rule 1 — Sensor grounding. PIR motion implies person location.
    (1, "PIR motion implies person location", """
        CONSTRUCT { ?p nsiot:isLocatedIn ?room }
        WHERE {
            ?p   a nsiot:Person .
            ?s   a nsiot:PIRMotionSensor .
            ?s   nsiot:isLocatedIn ?room .
            ?s   nsiot:hasState nsiot:StateMotionDetected .
        }
    """),

    # Rule 2 — Sensor grounding. Appliance plug ON implies person location.
    (2, "Appliance interaction implies person location", """
        CONSTRUCT { ?p nsiot:isLocatedIn ?room }
        WHERE {
            ?p    a nsiot:Person .
            ?plug a nsiot:SmartAppliancePlug .
            ?plug nsiot:isLocatedIn ?room .
            ?plug nsiot:hasState nsiot:StateON .
        }
    """),

    # Rule 3 — Validation. High-confidence Cook (MealPreparation) prediction
    # corroborated by person located in Kitchen.
    (3, "High-confidence Cook in Kitchen validation", """
        CONSTRUCT {
            ?p nsiot:performsActivity nsiot:MealPreparation .
            ?e a nsiot:ValidatedEvent .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:predictsActivity nsiot:MealPreparation .
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf > 0.85)
            ?p    nsiot:isLocatedIn ?room .
            ?room a nsiot:Kitchen .
        }
    """),

    # Rule 4 — Validation. High-confidence Eat prediction corroborated by
    # person located in Kitchen (CASAS Kyoto ADL eating happens in the
    # kitchen; the SAREF/AAL ontology's DiningArea is not instantiated).
    (4, "High-confidence Eat in Kitchen validation", """
        CONSTRUCT {
            ?p nsiot:performsActivity nsiot:Eating .
            ?e a nsiot:ValidatedEvent .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:predictsActivity nsiot:Eating .
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf > 0.80)
            ?p    nsiot:isLocatedIn ?room .
            ?room a nsiot:Kitchen .
        }
    """),

    # Rule 5 — AAL anomaly. High-confidence Cook predicted but kitchen
    # appliance plug NOT ON: unattended-cooking-style false alarm.
    (5, "Cook without kitchen appliance activation", """
        CONSTRUCT {
            ?e a nsiot:CriticalAnomaly .
            ?e nsiot:hasAlertType nsiot:UnattendedAppliance .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:predictsActivity nsiot:MealPreparation .
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf > 0.85)
            FILTER NOT EXISTS {
                ?plug a nsiot:SmartAppliancePlug .
                ?plug nsiot:isLocatedIn ?k .
                ?k    a nsiot:Kitchen .
                ?plug nsiot:hasState nsiot:StateON .
            }
        }
    """),

    # Rule 6 — AAL anomaly. Unattended fire hazard: kitchen plug ON while
    # person derived to be in a non-kitchen room.
    (6, "Unattended fire hazard (kitchen plug ON, person elsewhere)", """
        CONSTRUCT {
            ?e a nsiot:CriticalAnomaly .
            ?e nsiot:hasAlertType nsiot:UnattendedFireHazard .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?plug a nsiot:SmartAppliancePlug .
            ?plug nsiot:isLocatedIn ?k .
            ?k    a nsiot:Kitchen .
            ?plug nsiot:hasState nsiot:StateON .
            ?p    nsiot:isLocatedIn ?other .
            FILTER(?other != ?k)
        }
    """),

    # Rule 7 — AAL anomaly. Nocturnal kitchen activity (any of Cook, Eat,
    # Housekeeping predicted with high confidence during the NightTime
    # context) — common late-onset-dementia indicator.
    (7, "Nocturnal kitchen activity", """
        CONSTRUCT {
            ?e a nsiot:BehavioralAnomaly .
            ?e nsiot:hasAlertType nsiot:NocturnalActivity .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:hasTimeContext ?ctx .
            ?ctx  a nsiot:NightTimeContext .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:predictsActivity ?act .
            FILTER(?act IN (nsiot:MealPreparation, nsiot:Eating, nsiot:Housekeeping))
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf > 0.80)
        }
    """),

    # Rule 8 — Feedback trigger. Spatial hallucination: low-confidence
    # kitchen-activity prediction with NO supporting kitchen PIR motion in
    # the window.
    (8, "Spatial hallucination (kitchen activity without kitchen motion)", """
        CONSTRUCT {
            ?e a nsiot:FeedbackRequired .
            ?e nsiot:hasErrorType nsiot:FalsePositiveHallucination .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:predictsActivity ?act .
            FILTER(?act IN (nsiot:MealPreparation, nsiot:Eating, nsiot:Housekeeping))
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf < 0.80)
            FILTER NOT EXISTS {
                ?ks  a nsiot:PIRMotionSensor .
                ?ks  nsiot:isLocatedIn ?k .
                ?k   a nsiot:Kitchen .
                ?ks  nsiot:hasState nsiot:StateMotionDetected .
            }
        }
    """),

    # Rule 9 — Feedback trigger. Contextual mismatch: low-confidence
    # MealPreparation predicted but the derived person location is NOT
    # Kitchen.
    (9, "Contextual mismatch (Cook predicted, person not in Kitchen)", """
        CONSTRUCT {
            ?e a nsiot:FeedbackRequired .
            ?e nsiot:hasErrorType nsiot:ContextualMismatch .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:predictsActivity nsiot:MealPreparation .
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf < 0.85)
            ?p    nsiot:isLocatedIn ?room .
            FILTER NOT EXISTS {
                ?p nsiot:isLocatedIn ?k .
                ?k a nsiot:Kitchen .
            }
        }
    """),

    # Rule 10 — Feedback trigger. Low-margin top-1/top-2 prediction pair
    # *combined with* low absolute top-1 confidence. Requires the KG
    # builder to emit an alternative NeuralPrediction tagged
    # ``isAlternativePrediction true`` (top-2 softmax). The combined
    # filter (margin < 0.10 AND top-1 conf < 0.75) avoids over-flagging
    # confident-but-narrow predictions on well-trained models.
    (10, "Low-margin AND low-confidence top-2 disagreement", """
        CONSTRUCT {
            ?e a nsiot:FeedbackRequired .
            ?e nsiot:hasErrorType nsiot:MutuallyExclusiveActivities .
        }
        WHERE {
            ?e     a nsiot:Event .
            ?e     nsiot:isBasedOnPrediction ?pred1 .
            ?pred1 nsiot:isAlternativePrediction false .
            ?pred1 nsiot:predictsActivity ?act1 .
            ?pred1 nsiot:hasConfidenceScore ?conf1 .
            ?e     nsiot:isBasedOnPrediction ?pred2 .
            ?pred2 nsiot:isAlternativePrediction true .
            ?pred2 nsiot:predictsActivity ?act2 .
            ?pred2 nsiot:hasConfidenceScore ?conf2 .
            FILTER(?act1 != ?act2)
            FILTER((?conf1 - ?conf2) < 0.10)
            FILTER(?conf1 < 0.75)
        }
    """),

    # Rule 11 — Feedback trigger. Unsupported claim: low-confidence
    # prediction AND no person location derivable from any sensor.
    (11, "Unsupported claim (low conf + no sensor evidence)", """
        CONSTRUCT {
            ?e a nsiot:FeedbackRequired .
            ?e nsiot:hasErrorType nsiot:UnsupportedClaim .
        }
        WHERE {
            ?e    a nsiot:Event .
            ?e    nsiot:involvesPerson ?p .
            ?e    nsiot:isBasedOnPrediction ?pred .
            ?pred nsiot:hasConfidenceScore ?conf .
            FILTER(?conf < 0.75)
            FILTER NOT EXISTS {
                ?p nsiot:isLocatedIn ?_room .
            }
        }
    """),
]


def _ensure_room_self_types(g: Graph) -> None:
    """The KG builder uses room class IRIs (Kitchen, Bedroom, …) directly as
    the object of ``isLocatedIn`` (OWL 2 punning). Rule bodies that match
    ``?room a nsiot:Kitchen`` require an explicit individual-of-Kitchen
    triple. Inject those once before rule firing so rules 3–7, 9 can match.
    """
    for room in ("Kitchen", "Bedroom", "Bathroom", "LivingRoom"):
        room_uri = NSIOT[room]
        # Only assert if the room IRI is actually used somewhere.
        if (None, NSIOT.isLocatedIn, room_uri) in g and (room_uri, RDF.type, room_uri) not in g:
            g.add((room_uri, RDF.type, room_uri))


def fire_rules(g: Graph, *, max_iters: int = 10) -> Dict[str, Any]:
    """Fire all rules in a fixed-point loop. Mutates ``g`` in place.

    Returns a per-rule firing count and per-iteration deltas, useful for
    paper-side reporting of which rules contributed to the inference.
    """
    _ensure_room_self_types(g)
    rule_firings = {rid: 0 for rid, _, _ in RULES}
    iters: List[Dict[str, int]] = []
    total_added = 0

    for it in range(max_iters):
        new_this_iter = 0
        per_rule_this_iter: Dict[int, int] = {}
        for rid, _, query in RULES:
            try:
                triples = list(g.query(PREFIXES + query))
            except Exception as exc:  # rdflib SPARQL parse / runtime error
                log.warning("Rule %d query failed: %s", rid, exc)
                continue
            added_for_rule = 0
            for triple in triples:
                if len(triple) != 3:
                    continue
                if triple not in g:
                    g.add(triple)
                    added_for_rule += 1
                    new_this_iter += 1
            if added_for_rule:
                rule_firings[rid] += added_for_rule
                per_rule_this_iter[rid] = added_for_rule
        iters.append(per_rule_this_iter)
        total_added += new_this_iter
        if new_this_iter == 0:
            break

    return {
        "iterations": len(iters),
        "inferred_triples": total_added,
        "rule_firings": rule_firings,
        "per_iteration": iters,
    }


def _events_with_type(g: Graph, type_iri: URIRef) -> List[Dict[str, Any]]:
    """Return one entry per event individual carrying the given type."""
    out: List[Dict[str, Any]] = []
    for ev in g.subjects(RDF.type, type_iri):
        entry: Dict[str, Any] = {"uri": str(ev)}
        for _, _, alert in g.triples((ev, NSIOT.hasAlertType, None)):
            entry["alert_type"] = str(alert)
            break
        for _, _, err in g.triples((ev, NSIOT.hasErrorType, None)):
            entry["error_type"] = str(err)
            break
        for _, _, pred in g.triples((ev, NSIOT.isBasedOnPrediction, None)):
            for _, _, conf in g.triples((pred, NSIOT.hasConfidenceScore, None)):
                try:
                    entry["confidence"] = float(conf)
                except (TypeError, ValueError):
                    pass
                break
            break
        out.append(entry)
    return out


def _person_to_activities(g: Graph) -> List[Dict[str, Any]]:
    out = []
    for p, _, act in g.triples((None, NSIOT.performsActivity, None)):
        if (p, RDF.type, NSIOT.Person) in g:
            out.append({"person": str(p), "activity": str(act)})
    return out


def _person_to_locations(g: Graph) -> List[Dict[str, Any]]:
    out = []
    for p, _, room in g.triples((None, NSIOT.isLocatedIn, None)):
        if (p, RDF.type, NSIOT.Person) in g:
            out.append({"person": str(p), "location": str(room)})
    return out


def run_python_reasoning(populated_kg_path: Path) -> ReasoningResult:
    """Drop-in replacement for ``run_reasoning`` using the Python rule executor.

    Loads the populated KG via rdflib, fires the 11 SWRL rules to fixed
    point, and packages the resulting events into a :class:`ReasoningResult`.
    """
    result = ReasoningResult()
    t0 = time.time()

    g = Graph()
    g.parse(str(populated_kg_path), format="turtle")
    triples_before = len(g)

    fire_summary = fire_rules(g)
    log.info(
        "Python rule executor: %d iterations, %d new triples, firings=%s",
        fire_summary["iterations"],
        fire_summary["inferred_triples"],
        fire_summary["rule_firings"],
    )

    triples_after = len(g)
    result.num_inferred_triples = max(0, triples_after - triples_before)

    result.validated_events = _events_with_type(g, NSIOT.ValidatedEvent)
    result.critical_anomalies = _events_with_type(g, NSIOT.CriticalAnomaly)
    result.behavioral_anomalies = _events_with_type(g, NSIOT.BehavioralAnomaly)
    result.feedback_flags = _events_with_type(g, NSIOT.FeedbackRequired)
    result.inferred_activities = _person_to_activities(g)
    result.inferred_locations = _person_to_locations(g)
    result.reasoning_time_seconds = time.time() - t0
    result.inconsistencies = []

    log.info(
        "Python reasoning done in %.2fs: %d validated, %d critical, %d behavioral, %d feedback, %d inferred",
        result.reasoning_time_seconds,
        len(result.validated_events),
        len(result.critical_anomalies),
        len(result.behavioral_anomalies),
        len(result.feedback_flags),
        result.num_inferred_triples,
    )

    # Re-serialise so downstream steps (feedback loop's retract step) see
    # the inferred triples. Make the path absolute and ensure the parent
    # directory exists — defensive against any relative-path or CWD shift
    # between parse and serialize.
    out_path = Path(populated_kg_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")

    # Stash for callers that want the per-rule count
    result._rule_firings = fire_summary["rule_firings"]  # type: ignore[attr-defined]
    return result
