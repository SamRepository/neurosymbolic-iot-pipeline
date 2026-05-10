from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Structured output from the symbolic reasoner."""

    validated_events: List[Dict[str, Any]] = field(default_factory=list)
    critical_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    feedback_flags: List[Dict[str, Any]] = field(default_factory=list)
    inferred_activities: List[Dict[str, Any]] = field(default_factory=list)
    inferred_locations: List[Dict[str, Any]] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    num_inferred_triples: int = 0
    reasoning_time_seconds: float = 0.0


def _check_java() -> bool:
    """Check that Java is available (required by HermiT)."""
    return shutil.which("java") is not None


def load_ontology_owlready2(
    ontology_path: Path,
    populated_kg_path: Optional[Path] = None,
) -> Any:
    """
    Load ontology (+ optionally populated KG) into an owlready2 World.

    Returns the owlready2 ontology object.
    """
    import owlready2

    world = owlready2.World()

    # owlready2 does not support Turtle natively (only ntriples/rdfxml/owlxml).
    # Convert .ttl files to N-Triples in memory via rdflib first, then hand
    # the bytes to owlready2 as a fileobj. This also sidesteps Windows-path
    # quirks in owlready2's URI handling.
    import io
    from rdflib import Graph as _RdfGraph
    from rdflib.namespace import OWL as _OWL

    def _load_ttl(path_obj: Path) -> Any:
        abs_path = str(path_obj.resolve())
        if abs_path.lower().endswith(".ttl"):
            g = _RdfGraph()
            g.parse(abs_path, format="turtle")
            # Drop owl:imports — owlready2 would otherwise try to fetch
            # SOSA/SAREF/etc. from the network and parse them as NTriples,
            # which fails. The 11 SWRL rules we run do not depend on the
            # imported vocabularies' axioms.
            for s, _, o in list(g.triples((None, _OWL.imports, None))):
                g.remove((s, _OWL.imports, o))
            # Bind the namespaces SWRL rules use, so the RDF/XML serialiser
            # emits xmlns:nsiot=... etc. instead of auto-generated ns0:.
            g.bind("nsiot", "http://example.org/neuro-symbolic-iot#",
                   replace=True, override=True)
            g.bind("swrlb", "http://www.w3.org/2003/11/swrlb#",
                   replace=True, override=True)
            xml_bytes = g.serialize(format="xml", encoding="utf-8")
            fh = io.BytesIO(xml_bytes)
            return world.get_ontology(abs_path).load(fileobj=fh)
        with open(abs_path, "rb") as fh:
            return world.get_ontology(abs_path).load(fileobj=fh)

    onto = _load_ttl(Path(ontology_path))
    log.info("Loaded ontology: %s (%d classes)", onto.base_iri, len(list(onto.classes())))

    if populated_kg_path and Path(populated_kg_path).exists():
        _load_ttl(Path(populated_kg_path))
        log.info("Loaded populated KG: %s", populated_kg_path)

    return onto


def define_swrl_rules(onto: Any) -> List[Any]:
    """
    Programmatically define the 11 SWRL rules using owlready2 Imp().

    This is preferable to Turtle-encoded SWRL because owlready2's Imp() API
    handles variable binding and built-in atoms correctly.

    Note on prefixes: owlready2's SWRL parser resolves bare entity names
    against the active ontology (``with onto:``) and recognises swrlb
    built-ins (greaterThan, lessThan, …) as bare names. CURIE prefixes
    (``nsiot:`` / ``swrlb:``) are only resolved against namespaces emitted
    as ``xmlns:`` declarations during RDF/XML parsing, which our ontology
    does not provide via the Turtle → RDF/XML round-trip. We therefore
    strip the prefixes from each rule string before handing it to
    ``Imp.set_as_rule()``.
    """
    import re
    from owlready2 import Imp

    _PREFIX_RE = re.compile(r"\b(?:nsiot|swrlb):")

    # The ontology's base_iri ends with '...#NeuroSymbolicIoTOntology#' but
    # the SWRL entities (Person, Kitchen, isLocatedIn, ...) live in
    # 'http://example.org/neuro-symbolic-iot#'. Pass that namespace
    # explicitly to the rule parser so bare names resolve to the right IRIs.
    nsiot_ns = onto.get_namespace("http://example.org/neuro-symbolic-iot#")

    def _imp(rule: str) -> Any:
        r = Imp()
        r.set_as_rule(_PREFIX_RE.sub("", rule), namespaces=[nsiot_ns])
        return r

    rules = []

    with onto:
        # Rule 1: PIR motion -> person location
        r1 = _imp(
            "nsiot:Person(?p), nsiot:PIRMotionSensor(?s), "
            "nsiot:isLocatedIn(?s, ?room), nsiot:hasState(?s, nsiot:StateMotionDetected) "
            "-> nsiot:isLocatedIn(?p, ?room)"
        )
        rules.append(r1)

        # Rule 2: Appliance interaction -> person location
        r2 = _imp(
            "nsiot:Person(?p), nsiot:SmartAppliancePlug(?plug), "
            "nsiot:isLocatedIn(?plug, ?room), nsiot:hasState(?plug, nsiot:StateON) "
            "-> nsiot:isLocatedIn(?p, ?room)"
        )
        rules.append(r2)

        # Rule 3: High-confidence meal preparation validation
        r3 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsActivity(?pred, nsiot:MealPreparation), "
            "nsiot:hasConfidenceScore(?pred, ?conf), "
            "swrlb:greaterThan(?conf, 0.85), "
            "nsiot:isLocatedIn(?p, ?room), nsiot:Kitchen(?room) "
            "-> nsiot:performsActivity(?p, nsiot:MealPreparation), nsiot:ValidatedEvent(?e)"
        )
        rules.append(r3)

        # Rule 4: Posture-based sleeping validation
        r4 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsPosture(?pred, nsiot:Lying), "
            "nsiot:hasConfidenceScore(?pred, ?conf), "
            "swrlb:greaterThan(?conf, 0.80), "
            "nsiot:isLocatedIn(?p, ?room), nsiot:Bedroom(?room) "
            "-> nsiot:performsActivity(?p, nsiot:Sleeping), nsiot:ValidatedEvent(?e)"
        )
        rules.append(r4)

        # Rule 5: Critical fall detection (lying in bathroom)
        r5 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsPosture(?pred, nsiot:Lying), "
            "nsiot:hasConfidenceScore(?pred, ?conf), "
            "swrlb:greaterThan(?conf, 0.85), "
            "nsiot:isLocatedIn(?p, ?room), nsiot:Bathroom(?room) "
            "-> nsiot:CriticalAnomaly(?e), nsiot:hasAlertType(?e, nsiot:FallAlert)"
        )
        rules.append(r5)

        # Rule 6: Unattended fire hazard
        r6 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:SmartAppliancePlug(?plug), "
            "nsiot:isLocatedIn(?plug, ?k), nsiot:Kitchen(?k), "
            "nsiot:hasState(?plug, nsiot:StateON), "
            "nsiot:isLocatedIn(?p, ?b), nsiot:Bedroom(?b) "
            "-> nsiot:CriticalAnomaly(?e), nsiot:hasAlertType(?e, nsiot:UnattendedFireHazard)"
        )
        rules.append(r6)

        # Rule 7: Night wandering / sleep disturbance
        r7 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:NightTimeContext(?time), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsPosture(?pred, nsiot:Standing), "
            "nsiot:isLocatedIn(?p, ?room), nsiot:Kitchen(?room) "
            "-> nsiot:BehavioralAnomaly(?e), nsiot:hasAlertType(?e, nsiot:SleepDisturbance)"
        )
        rules.append(r7)

        # Rule 8: Spatial hallucination (low confidence + quiet room)
        r8 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsActivity(?pred, nsiot:Housekeeping), "
            "nsiot:hasConfidenceScore(?pred, ?conf), "
            "swrlb:lessThan(?conf, 0.70), "
            "nsiot:isLocatedIn(?p, ?room), "
            "nsiot:hasState(?sensor, nsiot:StateQuiet), nsiot:isLocatedIn(?sensor, ?room) "
            "-> nsiot:FeedbackRequired(?e), nsiot:hasErrorType(?e, nsiot:FalsePositiveHallucination)"
        )
        rules.append(r8)

        # Rule 9: Contextual hallucination (sleeping in living room with TV on)
        r9 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsActivity(?pred, nsiot:Sleeping), "
            "nsiot:isLocatedIn(?p, ?room), nsiot:LivingRoom(?room), "
            "nsiot:SmartAppliancePlug(?tv), nsiot:hasState(?tv, nsiot:StateON), "
            "nsiot:isLocatedIn(?tv, ?room) "
            "-> nsiot:FeedbackRequired(?e), nsiot:hasErrorType(?e, nsiot:ContextualMismatch)"
        )
        rules.append(r9)

        # Rule 10: Mutually exclusive predictions
        r10 = _imp(
            "nsiot:Event(?e), "
            "nsiot:isBasedOnPrediction(?e, ?pred1), "
            "nsiot:predictsActivity(?pred1, nsiot:PersonalHygiene), "
            "nsiot:isBasedOnPrediction(?e, ?pred2), "
            "nsiot:predictsActivity(?pred2, nsiot:MealPreparation) "
            "-> nsiot:FeedbackRequired(?e), nsiot:hasErrorType(?e, nsiot:MutuallyExclusiveActivities)"
        )
        rules.append(r10)

        # Rule 11: Posture-based fall detection (lying + quiet sensors)
        r11 = _imp(
            "nsiot:Event(?e), nsiot:involvesPerson(?e, ?p), "
            "nsiot:isBasedOnPrediction(?e, ?pred), "
            "nsiot:predictsPosture(?pred, nsiot:Lying), "
            "nsiot:hasConfidenceScore(?pred, ?conf), "
            "swrlb:greaterThan(?conf, 0.80), "
            "nsiot:isLocatedIn(?p, ?room), "
            "nsiot:PIRMotionSensor(?s), nsiot:isLocatedIn(?s, ?room), "
            "nsiot:hasState(?s, nsiot:StateQuiet) "
            "-> nsiot:CriticalAnomaly(?e), nsiot:hasAlertType(?e, nsiot:FallAlert)"
        )
        rules.append(r11)

    log.info("Defined %d SWRL rules", len(rules))
    return rules


def run_reasoning(
    ontology_path: Path,
    populated_kg_path: Path,
    *,
    cfg: Dict[str, Any],
) -> ReasoningResult:
    """
    Full reasoning pipeline:
    1. Load ontology + populated KG into owlready2
    2. Define SWRL rules via Imp()
    3. Run sync_reasoner_hermit()
    4. Query inferred facts
    5. Package into ReasoningResult
    """
    # Two backends are supported:
    #   * "python" (default) — translates each of the 11 SWRL rules into a
    #     SPARQL CONSTRUCT and runs them in a fixed-point loop on rdflib.
    #     This is the only backend that actually executes the rule bodies;
    #     owlready2's wrappers around HermiT/Pellet only do OWL-DL
    #     classification and do not fire SWRL rules.
    #   * "hermit" / "pellet" — kept for ontology-validation use-cases.
    backend = str(cfg.get("reasoning", {}).get("rule_executor", "python")).lower()
    if backend == "python":
        from neurosymbolic_iot.reasoning_feedback.reasoning.rule_executor import (
            run_python_reasoning,
        )
        return run_python_reasoning(populated_kg_path)

    import owlready2

    if not _check_java():
        log.error("Java not found — HermiT reasoner requires Java. Install JRE/JDK.")
        return ReasoningResult(inconsistencies=["Java not found; reasoning skipped."])

    result = ReasoningResult()
    t0 = time.time()

    onto = load_ontology_owlready2(ontology_path, populated_kg_path)
    triples_before = len(list(onto.world.sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")))

    define_swrl_rules(onto)

    # Pellet is selected because HermiT does not natively evaluate SWRL
    # rule bodies that mix data-property comparisons (swrlb:greaterThan,
    # swrlb:lessThan) with object-property chains. Pellet (bundled with
    # owlready2) is SWRL-complete and runs the full 11-rule programme.
    reasoner_name = str(cfg.get("reasoning", {}).get("reasoner", "pellet")).lower()
    try:
        with onto:
            if reasoner_name == "hermit":
                owlready2.sync_reasoner_hermit(infer_property_values=True)
                log.info("HermiT reasoning completed successfully.")
            else:
                owlready2.sync_reasoner_pellet(
                    infer_property_values=True,
                    infer_data_property_values=True,
                )
                log.info("Pellet reasoning completed successfully.")
    except owlready2.OwlReadyInconsistentOntologyError as exc:
        result.inconsistencies.append(f"Ontology inconsistency: {exc}")
        log.warning("Ontology inconsistency detected: %s", exc)

    triples_after = len(list(onto.world.sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")))
    result.num_inferred_triples = max(0, triples_after - triples_before)

    # Extract inferred facts
    nsiot_ns = onto.get_namespace("http://example.org/neuro-symbolic-iot#")

    result.validated_events = _extract_typed_events(nsiot_ns, "ValidatedEvent")
    result.critical_anomalies = _extract_typed_events(nsiot_ns, "CriticalAnomaly")
    result.behavioral_anomalies = _extract_typed_events(nsiot_ns, "BehavioralAnomaly")
    result.feedback_flags = _extract_typed_events(nsiot_ns, "FeedbackRequired")
    result.inferred_activities = _extract_inferred_activities(nsiot_ns)
    result.inferred_locations = _extract_inferred_locations(nsiot_ns)

    result.reasoning_time_seconds = time.time() - t0
    log.info(
        "Reasoning done in %.2fs: %d validated, %d critical, %d behavioral, %d feedback, %d inferred triples",
        result.reasoning_time_seconds,
        len(result.validated_events),
        len(result.critical_anomalies),
        len(result.behavioral_anomalies),
        len(result.feedback_flags),
        result.num_inferred_triples,
    )
    return result


def _extract_typed_events(nsiot_ns: Any, class_name: str) -> List[Dict[str, Any]]:
    """Extract all individuals of a given Event subclass."""
    results = []
    cls = getattr(nsiot_ns, class_name, None)
    if cls is None:
        return results
    for individual in cls.instances():
        entry: Dict[str, Any] = {"uri": str(individual.iri)}
        if hasattr(individual, "hasAlertType") and individual.hasAlertType:
            entry["alert_type"] = str(individual.hasAlertType[0].iri) if individual.hasAlertType else None
        if hasattr(individual, "hasErrorType") and individual.hasErrorType:
            entry["error_type"] = str(individual.hasErrorType[0].iri) if individual.hasErrorType else None
        if hasattr(individual, "isBasedOnPrediction") and individual.isBasedOnPrediction:
            pred = individual.isBasedOnPrediction[0] if individual.isBasedOnPrediction else None
            if pred and hasattr(pred, "hasConfidenceScore") and pred.hasConfidenceScore:
                entry["confidence"] = float(pred.hasConfidenceScore[0])
        results.append(entry)
    return results


def _extract_inferred_activities(nsiot_ns: Any) -> List[Dict[str, Any]]:
    """Extract person -> activity from performsActivity."""
    results = []
    person_cls = getattr(nsiot_ns, "Person", None)
    if person_cls is None:
        return results
    for person in person_cls.instances():
        if hasattr(person, "performsActivity") and person.performsActivity:
            for act in person.performsActivity:
                results.append({"person": str(person.iri), "activity": str(act.iri)})
    return results


def _extract_inferred_locations(nsiot_ns: Any) -> List[Dict[str, Any]]:
    """Extract person -> room from isLocatedIn."""
    results = []
    person_cls = getattr(nsiot_ns, "Person", None)
    if person_cls is None:
        return results
    for person in person_cls.instances():
        if hasattr(person, "isLocatedIn") and person.isLocatedIn:
            for loc in person.isLocatedIn if isinstance(person.isLocatedIn, list) else [person.isLocatedIn]:
                results.append({"person": str(person.iri), "location": str(loc.iri)})
    return results
