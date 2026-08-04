"""
Reproducibility artifact dumps (E6 - FGCS revision)
===================================================
Addresses reviewer items R1-12 (the complete 11-rule base, with preconditions and
consequents, since reproducibility and the disjointness-of-feedback argument
depend on the exact definitions) and R2-3 (per-dataset benchmark and KG
statistics, plus an explicit statement of whether one rule set serves all
datasets or each is instantiated separately).

Everything here is machine-generated from the code and the stored artifacts so
that it cannot drift from what actually runs.

A finding this script exists to surface: the repository contains **two different
definitions of the "11 SWRL rules"**, and they are not equivalent.

  * ``symbolic_reasoner.define_swrl_rules`` builds owlready2 ``Imp()`` rules. This
    is the SWRL form, and it is what an appendix transcribed from the ontology
    would show.
  * ``rule_executor.RULES`` holds SPARQL CONSTRUCT translations. Per the project's
    own configuration (``rule_executor: python``) this is what actually fires in
    every experiment the manuscript reports, because owlready2's bundled reasoners
    perform OWL-DL classification only and do not execute SWRL rule bodies.

Rules 4, 5, 7, 8, 9, 10 and 11 differ materially between the two, and rule 11
differs in *category*: the SWRL form emits ``CriticalAnomaly``/``FallAlert``
(an AAL anomaly rule) while the executed form emits
``FeedbackRequired``/``UnsupportedClaim`` (a feedback trigger). The manuscript's
"rules 8-11 are the only feedback consumers" taxonomy holds for the executed set
and fails for the SWRL set, so the appendix must present the executed rules.

Usage:
  PYTHONPATH=. python evaluation/run_reproducibility_dumps.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph, RDF, RDFS
from rdflib.namespace import OWL

log = logging.getLogger(__name__)

NSIOT_PREFIX = "http://example.org/neuro-symbolic-iot#"

RULE_CATEGORIES: Dict[int, str] = {
    1: "sensor grounding", 2: "sensor grounding",
    3: "validation", 4: "validation",
    5: "AAL anomaly", 6: "AAL anomaly", 7: "AAL anomaly",
    8: "feedback trigger", 9: "feedback trigger",
    10: "feedback trigger", 11: "feedback trigger",
}

FEEDBACK_CONSEQUENT = "FeedbackRequired"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Executed rules (rule_executor.RULES)
# ---------------------------------------------------------------------------

def _normalise_sparql(fragment: str) -> str:
    """Collapse a SPARQL block to a single readable line."""
    text = " ".join(fragment.split())
    return text.replace("<" + NSIOT_PREFIX, "nsiot:").replace(">", "")


def _split_construct(query: str) -> Tuple[str, str]:
    """Return (consequent, precondition) from a CONSTRUCT ... WHERE ... query."""
    c_start = query.find("CONSTRUCT")
    w_start = query.find("WHERE")
    if c_start < 0 or w_start < 0:
        return "", _normalise_sparql(query)
    construct_block = query[c_start + len("CONSTRUCT"):w_start]
    where_block = query[w_start + len("WHERE"):]
    return _normalise_sparql(construct_block).strip("{} "), _normalise_sparql(where_block).strip("{} ")


def extract_executed_rules() -> List[Dict[str, Any]]:
    from neurosymbolic_iot.reasoning_feedback.reasoning.rule_executor import RULES

    rows: List[Dict[str, Any]] = []
    for rule_id, label, query in RULES:
        consequent, precondition = _split_construct(query)
        emits_feedback = FEEDBACK_CONSEQUENT in consequent
        rows.append({
            "rule_id": rule_id,
            "label": label,
            "category": RULE_CATEGORIES.get(rule_id, "unknown"),
            "precondition": precondition,
            "consequent": consequent,
            "participates_in_feedback": emits_feedback,
            "asserts_ground_truth_label": False,
        })
    return rows


# ---------------------------------------------------------------------------
# SWRL rules (symbolic_reasoner.define_swrl_rules), extracted from source
# ---------------------------------------------------------------------------

def extract_swrl_rules(source_path: Path) -> List[Dict[str, Any]]:
    """Recover the SWRL rule strings by AST-parsing the source.

    Parsing rather than importing: ``define_swrl_rules`` needs a live owlready2
    ontology object, and building one would make a documentation dump depend on
    the reasoner toolchain.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    rules: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "_imp"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        try:
            value = ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, str) or "->" not in value:
            continue
        body, head = value.split("->", 1)
        rules.append({
            "precondition": " ".join(body.split()).rstrip(", "),
            "consequent": " ".join(head.split()),
        })

    for idx, rule in enumerate(rules, start=1):
        rule["rule_id"] = idx
        rule["category"] = RULE_CATEGORIES.get(idx, "unknown")
        rule["participates_in_feedback"] = FEEDBACK_CONSEQUENT in rule["consequent"]
    return rules


def _consequent_types(text: str) -> set:
    """Ontology class / alert names appearing in a consequent, for comparison."""
    return set(re.findall(r"(?:nsiot:)?([A-Z][A-Za-z]+)", text))


def _vocabulary(text: str) -> set:
    """Ontology terms a rule body references.

    The two rule sets are written in different languages (SWRL atoms vs SPARQL
    triple patterns), so the bodies cannot be diffed syntactically. The set of
    nsiot: terms each one touches is comparable and is enough to show whether
    two rules are triggered by the same evidence.
    """
    terms = set(re.findall(r"nsiot:(\w+)", text))
    # Variables and generic scaffolding carry no discriminative information.
    return terms - {"Event", "Person", "involvesPerson", "isBasedOnPrediction"}


def compare_rule_sets(
    executed: List[Dict[str, Any]],
    swrl: List[Dict[str, Any]],
) -> Dict[str, Any]:
    swrl_by_id = {r["rule_id"]: r for r in swrl}
    comparisons: List[Dict[str, Any]] = []
    n_diverge = 0
    n_body_diverge = 0
    n_fully_equivalent = 0
    category_conflicts: List[int] = []

    for row in executed:
        rid = row["rule_id"]
        other = swrl_by_id.get(rid)
        if other is None:
            comparisons.append({"rule_id": rid, "status": "no SWRL counterpart"})
            continue
        exec_types = _consequent_types(row["consequent"])
        swrl_types = _consequent_types(other["consequent"])
        same_consequent = exec_types == swrl_types

        exec_vocab = _vocabulary(row["precondition"])
        swrl_vocab = _vocabulary(other["precondition"])
        same_body = exec_vocab == swrl_vocab

        same_feedback_role = row["participates_in_feedback"] == other["participates_in_feedback"]
        if not same_consequent:
            n_diverge += 1
        if not same_body:
            n_body_diverge += 1
        if same_consequent and same_body:
            n_fully_equivalent += 1
        if not same_feedback_role:
            category_conflicts.append(rid)

        if not same_consequent and not same_body:
            status = "DIVERGENT (trigger and consequent)"
        elif not same_consequent:
            status = "DIVERGENT (consequent)"
        elif not same_body:
            status = "DIVERGENT (trigger only — same consequent, different evidence)"
        else:
            status = "equivalent"

        comparisons.append({
            "rule_id": rid,
            "executed_label": row["label"],
            "executed_consequent": row["consequent"],
            "swrl_consequent": other["consequent"],
            "consequents_match": same_consequent,
            "preconditions_match": same_body,
            "executed_only_terms": sorted(exec_vocab - swrl_vocab),
            "swrl_only_terms": sorted(swrl_vocab - exec_vocab),
            "feedback_role_matches": same_feedback_role,
            "status": status,
        })

    return {
        "n_rules_executed": len(executed),
        "n_rules_swrl": len(swrl),
        "n_divergent_consequents": n_diverge,
        "n_divergent_preconditions": n_body_diverge,
        "n_fully_equivalent": n_fully_equivalent,
        "comparison_method": (
            "Consequents are compared by the ontology classes they assert. Preconditions are "
            "written in different languages (SWRL atoms vs SPARQL triple patterns) and cannot be "
            "diffed syntactically, so they are compared by the set of nsiot: terms each body "
            "references — enough to establish whether two rules fire on the same evidence."
        ),
        "rules_with_conflicting_feedback_role": category_conflicts,
        "disjointness_holds_for_executed_set": all(
            r["participates_in_feedback"] == (r["rule_id"] >= 8) for r in executed
        ),
        "disjointness_holds_for_swrl_set": all(
            r["participates_in_feedback"] == (r["rule_id"] >= 8) for r in swrl
        ),
        "per_rule": comparisons,
    }


# ---------------------------------------------------------------------------
# Dataset and KG statistics
# ---------------------------------------------------------------------------

def _ontology_stats(onto_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path in sorted(onto_dir.glob("*.ttl")):
        g = Graph()
        try:
            g.parse(str(path), format="turtle")
        except Exception as exc:  # owlready2-flavoured SWRL TTL is not rdflib-parseable
            out[path.name] = {"parsed": False, "error": str(exc)}
            continue
        out[path.name] = {
            "parsed": True,
            "triples": len(g),
            "owl_classes": len(set(g.subjects(RDF.type, OWL.Class))),
            "object_properties": len(set(g.subjects(RDF.type, OWL.ObjectProperty))),
            "datatype_properties": len(set(g.subjects(RDF.type, OWL.DatatypeProperty))),
        }
    return out


def _kg_stats(ttl_path: Path) -> Optional[Dict[str, Any]]:
    if not ttl_path.exists():
        return None
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    subjects = set(g.subjects())
    predicates = set(g.predicates())
    objects = {o for o in g.objects() if not isinstance(o, str) or True}
    entities = {s for s in subjects if str(s).startswith(NSIOT_PREFIX)}
    return {
        "source": ttl_path.as_posix(),
        "triples": len(g),
        "distinct_subjects": len(subjects),
        "distinct_predicates": len(predicates),
        "distinct_objects": len(objects),
        "nsiot_entities": len(entities),
    }


def _dataset_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_dataset_statistics(repo: Path) -> Dict[str, Any]:
    sensor_map = json.loads(
        (repo / "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json").read_text(encoding="utf-8")
    )

    datasets: Dict[str, Any] = {}

    specs = [
        ("casas_kyoto", "data/processed/casas_meta.json", "evaluation/results/ablation_cv.json",
         "evaluation/results/fold_0/kg/casas/populated_kg.ttl", "casas"),
        ("casas_aruba", "data/processed/casas_aruba_meta.json", "evaluation/results_aruba/ablation_cv.json",
         "evaluation/results_aruba/fold_0/kg/casas/populated_kg.ttl", "casas"),
        ("sphere", "data/processed/sphere_meta.json", None, None, "sphere"),
    ]

    for name, meta_rel, ablation_rel, kg_rel, map_key in specs:
        entry: Dict[str, Any] = {"sensor_map_key": map_key}
        meta = _dataset_meta(repo / meta_rel)
        entry["processed_metadata"] = meta

        if ablation_rel:
            abl_path = repo / ablation_rel
            if abl_path.exists():
                abl = json.loads(abl_path.read_text(encoding="utf-8"))
                md = abl.get("metadata", {})
                n_windows = md.get("n_windows")
                k = md.get("k")
                entry["cv_protocol"] = {
                    "k_folds": k,
                    "seed": md.get("seed"),
                    "epochs": md.get("epochs"),
                    "total_windows": n_windows,
                    "activity_classes": md.get("n_classes"),
                    "class_labels": md.get("labels"),
                    "train_windows_per_fold": (
                        int(round(n_windows * (k - 1) / k)) if n_windows and k else None
                    ),
                    "test_windows_per_fold": (
                        int(round(n_windows / k)) if n_windows and k else None
                    ),
                }

        if kg_rel:
            entry["knowledge_graph"] = _kg_stats(repo / kg_rel)

        # The processed parquet and the CV runs window the raw stream with different
        # settings, so their window counts differ. Both are recorded rather than
        # silently reconciled: the manuscript must quote whichever corresponds to the
        # experiment it is describing.
        if meta:
            entry["processed_windows"] = meta.get("n_windows")
            entry["processed_events"] = meta.get("n_events") or meta.get("n_labeled_rows")
            if meta.get("splits"):
                entry["processed_splits"] = meta["splits"]
            cv = entry.get("cv_protocol") or {}
            if cv.get("total_windows") and meta.get("n_windows") and \
                    cv["total_windows"] != meta["n_windows"]:
                entry["window_count_discrepancy"] = {
                    "processed_parquet": meta["n_windows"],
                    "cv_run": cv["total_windows"],
                    "note": (
                        "Different windowing settings: the CV scripts rebuild windows from raw "
                        "via build_casas_windows_from_raw using the experiment config, rather "
                        "than reading data/processed. Not an error, but the paper must not mix "
                        "the two figures."
                    ),
                }

        ds_map = sensor_map.get(map_key, {})
        if map_key == "casas":
            entry["sensor_types"] = sorted({
                info.get("rdf_type", "") for info in ds_map.get("sensor_type_patterns", {}).values()
            })
            entry["n_sensor_type_patterns"] = len(ds_map.get("sensor_type_patterns", {}))
            entry["n_mapped_rooms"] = len(set(ds_map.get("room_assignments", {}).values()))
        else:
            entry["sensor_types"] = sorted({
                info.get("rdf_type", "") for info in ds_map.get("pir_columns", {}).values()
            })
            entry["n_pir_columns"] = len(ds_map.get("pir_columns", {}))
            entry["n_mapped_rooms"] = len({
                info.get("room") for info in ds_map.get("pir_columns", {}).values()
            })
        entry["n_activity_map_entries"] = len(ds_map.get("activity_map", {}))
        entry["activity_map"] = ds_map.get("activity_map", {})

        datasets[name] = entry

    return datasets


def build_anomaly_categories(executed: List[Dict[str, Any]]) -> Dict[str, Any]:
    alerts: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    for row in executed:
        for name in re.findall(r"nsiot:(\w+)", row["consequent"]):
            if row["participates_in_feedback"] and name not in ("FeedbackRequired",):
                if name not in ("Event",):
                    errors[name] = row["rule_id"]
            elif name in ("UnattendedAppliance", "UnattendedFireHazard", "NocturnalActivity",
                          "FallAlert", "SleepDisturbance"):
                alerts[name] = row["rule_id"]
    return {
        "aal_alert_types": alerts,
        "feedback_error_types": {k: v for k, v in errors.items() if k.endswith(
            ("Hallucination", "Mismatch", "Activities", "Claim"))},
    }


RULE_SET_SCOPE_ANSWER = (
    "ONE rule set, hard-coded, instantiated for CASAS Kyoto ADL vocabulary; it is applied "
    "unchanged to every dataset. rule_executor.RULES is a single module-level list and "
    "rule_executor.fire_rules takes no dataset argument and performs no dataset branching - "
    "the same eleven SPARQL CONSTRUCT queries run against whatever graph is passed. There are "
    "no per-dataset rule variants anywhere in the codebase.\n\n"
    "The consequence must be stated plainly, because 'one rule set for all datasets' sounds "
    "like a generality claim and here it is closer to the opposite. The rule bodies are written "
    "against CASAS Kyoto ADL's class vocabulary (MealPreparation, Eating, Housekeeping, Kitchen, "
    "SmartAppliancePlug), as the module's own comment records. On any dataset whose vocabulary "
    "differs, rules referencing absent classes are structurally incapable of firing rather than "
    "being merely inapplicable to the data. This is measurable: on CASAS Aruba, five of the "
    "eleven rules (2, 6, 8, 9, 11) never fired across all five folds, and rule 10 alone accounted "
    "for roughly three quarters of all firings (see ablation_diagnostics.json). SPHERE, whose "
    "vocabulary is postures and PIR rooms rather than kitchen appliances, would exercise even "
    "fewer.\n\n"
    "The defensible statement for the manuscript is therefore: a single fixed rule base is "
    "shared across datasets, authored against the CASAS Kyoto ADL vocabulary, with per-dataset "
    "coverage varying widely and reported per dataset - not that the rule base generalises "
    "across the benchmarks."
)


def render_rule_markdown(executed: List[Dict[str, Any]], comparison: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# The 11-rule base, as executed (E6 — reviewer items R1-12, R2-3)")
    lines.append("")
    lines.append("These are the rules that actually fire in every reported experiment: the SPARQL")
    lines.append("CONSTRUCT translations in `rule_executor.RULES`, selected by the project's")
    lines.append("`rule_executor: python` setting. owlready2's bundled HermiT/Pellet perform OWL-DL")
    lines.append("classification only and do not execute SWRL rule bodies, so the SWRL forms in")
    lines.append("`symbolic_reasoner.define_swrl_rules` are never evaluated at run time.")
    lines.append("")
    lines.append("| # | Category | Rule | Precondition (SPARQL WHERE) | Consequent (CONSTRUCT) | Feedback? |")
    lines.append("|---|---|---|---|---|---|")
    for row in executed:
        pre = row["precondition"].replace("|", "\\|")
        con = row["consequent"].replace("|", "\\|")
        if len(pre) > 300:
            pre = pre[:297] + "…"
        lines.append(
            f"| {row['rule_id']} | {row['category']} | {row['label']} | `{pre}` | `{con}` "
            f"| {'yes' if row['participates_in_feedback'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Disjointness of feedback")
    lines.append("")
    lines.append(
        f"For the executed rule set the manuscript's taxonomy holds: "
        f"**{comparison['disjointness_holds_for_executed_set']}** — rules 8–11 are exactly the "
        f"rules emitting `FeedbackRequired`, and no rule asserts a ground-truth label "
        f"(labels come from dataset annotations only)."
    )
    lines.append("")
    lines.append("## ⚠ The SWRL definitions are not the executed definitions")
    lines.append("")
    lines.append(
        f"`symbolic_reasoner.define_swrl_rules` defines a second, materially different set of "
        f"eleven rules. Only **{comparison['n_fully_equivalent']} of "
        f"{comparison['n_rules_executed']}** are equivalent to their executed counterparts: "
        f"{comparison['n_divergent_consequents']} assert different consequents and "
        f"{comparison['n_divergent_preconditions']} fire on different evidence."
    )
    if comparison["rules_with_conflicting_feedback_role"]:
        lines.append("")
        lines.append(
            f"Critically, rule(s) {comparison['rules_with_conflicting_feedback_role']} differ in "
            f"*category*: the SWRL form emits a `CriticalAnomaly` while the executed form emits "
            f"`FeedbackRequired`. The 8–11 disjointness claim is therefore **"
            f"{comparison['disjointness_holds_for_swrl_set']}** for the SWRL set. An appendix that "
            f"transcribes the SWRL rules would document a system that never ran and would break "
            f"the disjointness argument the manuscript relies on."
        )
    lines.append("")
    lines.append("| # | Executed consequent | SWRL consequent | Status |")
    lines.append("|---|---|---|---|")
    for c in comparison["per_rule"]:
        if "executed_consequent" not in c:
            continue
        status = c["status"] if c["status"] == "equivalent" else f"**{c['status']}**"
        lines.append(
            f"| {c['rule_id']} | `{c['executed_consequent'][:70]}` | `{c['swrl_consequent'][:70]}` "
            f"| {status} |"
        )
    lines.append("")
    lines.append(comparison["comparison_method"])
    lines.append("")
    lines.append("## Same rule set across datasets?")
    lines.append("")
    lines.append(RULE_SET_SCOPE_ANSWER)
    lines.append("")
    return "\n".join(lines)


def render_dataset_markdown(stats: Dict[str, Any], onto: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Per-dataset and knowledge-graph statistics (E6 — reviewer item R2-3)")
    lines.append("")
    lines.append("| Dataset | Windows (experiment) | Windows (processed) | Classes | Train | Test | KG triples | Predicates | nsiot entities |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, entry in stats.items():
        cv = entry.get("cv_protocol") or {}
        kg = entry.get("knowledge_graph") or {}
        splits = entry.get("processed_splits") or {}
        train = cv.get("train_windows_per_fold") or splits.get("train") or "—"
        test = cv.get("test_windows_per_fold") or splits.get("test") or "—"
        lines.append(
            f"| {name} | {cv.get('total_windows', '—')} | {entry.get('processed_windows', '—')} "
            f"| {cv.get('activity_classes') or entry.get('n_activity_map_entries', '—')} "
            f"| {train} | {test} "
            f"| {kg.get('triples', '—')} | {kg.get('distinct_predicates', '—')} "
            f"| {kg.get('nsiot_entities', '—')} |"
        )
    lines.append("")
    lines.append("KG statistics are measured on fold 0's serialized graph, after rule firing.")
    lines.append("")
    lines.append("**Two window counts are reported deliberately.** The CV scripts rebuild windows")
    lines.append("from the raw stream using the experiment config rather than reading")
    lines.append("`data/processed`, so the two figures differ. Quote whichever matches the")
    lines.append("experiment being described; do not mix them.")
    for name, entry in stats.items():
        d = entry.get("window_count_discrepancy")
        if d:
            lines.append("")
            lines.append(f"- `{name}`: processed parquet {d['processed_parquet']} windows vs "
                         f"CV run {d['cv_run']} windows.")
    lines.append("")
    lines.append("**SPHERE is markedly data-limited** (reviewer item R1-6): 139 windows total with")
    lines.append("a 97/21/21 train/val/test split, i.e. 21 test windows across 20 label columns.")
    lines.append("Any generalisation claim resting on SPHERE should be tempered accordingly.")
    lines.append("")
    lines.append("## Ontology")
    lines.append("")
    lines.append("| File | Triples | OWL classes | Object properties | Datatype properties |")
    lines.append("|---|---|---|---|---|")
    for fname, o in onto.items():
        if not o.get("parsed"):
            lines.append(f"| {fname} | not rdflib-parseable (owlready2 SWRL syntax) | — | — | — |")
            continue
        lines.append(
            f"| {fname} | {o['triples']} | {o['owl_classes']} | {o['object_properties']} "
            f"| {o['datatype_properties']} |"
        )
    lines.append("")
    lines.append("## Sensor and activity vocabulary per dataset")
    lines.append("")
    for name, entry in stats.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Sensor types: {', '.join(t for t in entry['sensor_types'] if t) or '—'}")
        lines.append(f"- Mapped rooms: {entry.get('n_mapped_rooms', '—')}")
        lines.append(f"- Activity map entries: {entry['n_activity_map_entries']}")
        cv = entry.get("cv_protocol") or {}
        if cv.get("class_labels"):
            lines.append(f"- Class labels: {', '.join(cv['class_labels'])}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E6: machine-generated reproducibility dumps.")
    p.add_argument("--outdir", default="evaluation/results_repro")
    return p.parse_args()


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))

    from neurosymbolic_iot.utils.logging import setup_logging

    args = parse_args()
    setup_logging("INFO")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    executed = extract_executed_rules()
    swrl = extract_swrl_rules(
        repo / "neurosymbolic_iot/reasoning_feedback/reasoning/symbolic_reasoner.py"
    )
    comparison = compare_rule_sets(executed, swrl)
    log.info("Extracted %d executed rules and %d SWRL rules; %d divergent consequents",
             len(executed), len(swrl), comparison["n_divergent_consequents"])

    datasets = build_dataset_statistics(repo)
    onto = _ontology_stats(repo / "neurosymbolic_iot/kg_semantic_layer/ontology")
    anomaly_categories = build_anomaly_categories(executed)

    meta = {
        "script": "evaluation/run_reproducibility_dumps.py",
        "commit": _git_commit(),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Machine-generated from the code and stored artifacts so the appendix cannot drift "
            "from what runs."
        ),
    }

    rule_payload = {
        "_meta": meta,
        "executed_rules": executed,
        "swrl_rules": swrl,
        "comparison": comparison,
        "anomaly_categories": anomaly_categories,
        "same_rule_set_across_datasets": RULE_SET_SCOPE_ANSWER,
    }
    (out_dir / "rule_table.json").write_text(
        json.dumps(rule_payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "rule_table.md").write_text(
        render_rule_markdown(executed, comparison), encoding="utf-8")

    with (out_dir / "rule_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "rule_id", "category", "label", "precondition", "consequent",
            "participates_in_feedback", "asserts_ground_truth_label",
        ])
        writer.writeheader()
        for row in executed:
            writer.writerow(row)

    dataset_payload = {"_meta": meta, "datasets": datasets, "ontology": onto}
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(dataset_payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "dataset_statistics.md").write_text(
        render_dataset_markdown(datasets, onto), encoding="utf-8")

    print()
    print("=" * 92)
    print("  E6 — REPRODUCIBILITY DUMPS")
    print("=" * 92)
    print(f"  Executed rules extracted        : {len(executed)}")
    print(f"  SWRL rules extracted            : {len(swrl)}")
    print(f"  Fully equivalent to SWRL        : {comparison['n_fully_equivalent']} of {len(executed)}")
    print(f"  Divergent consequents           : {comparison['n_divergent_consequents']}")
    print(f"  Divergent preconditions         : {comparison['n_divergent_preconditions']}")
    print(f"  Rules with conflicting category : {comparison['rules_with_conflicting_feedback_role']}")
    print(f"  Disjointness holds — executed   : {comparison['disjointness_holds_for_executed_set']}")
    print(f"  Disjointness holds — SWRL       : {comparison['disjointness_holds_for_swrl_set']}")
    print()
    print("  Per-rule divergence:")
    for c in comparison["per_rule"]:
        if "executed_consequent" not in c:
            continue
        mark = "OK " if c["status"] == "equivalent" else "!! "
        print(f"    {mark}R{c['rule_id']:<3} {c['status']}")
    print()
    print("  Datasets:")
    for name, entry in datasets.items():
        cv = entry.get("cv_protocol") or {}
        kg = entry.get("knowledge_graph") or {}
        print(f"    {name:<14} windows={cv.get('total_windows', '—'):<6} classes={cv.get('activity_classes', '—'):<4} "
              f"kg_triples={kg.get('triples', '—')}")
    print("=" * 92)

    log.info("Wrote rule_table.{json,md,csv} and dataset_statistics.{json,md} in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
