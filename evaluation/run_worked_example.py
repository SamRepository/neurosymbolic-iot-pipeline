"""
Worked end-to-end example (E10 - FGCS revision)
===============================================
Addresses reviewer item R2-8: export one concrete window from a benchmark, end to
end - raw sensor events, neural prediction and confidence, the RDF triples
emitted, the rule firings, and the final decision with its explanation trace.

The example is a **real window from the published Aruba run** (commit f5c6cd3),
not a constructed illustration. Every stage is read from the stored artifacts:
raw events from a deterministic rebuild of the window table, the prediction and
its triples from the fold's serialized KG, and the rule outcomes from firing the
rules themselves.

Rule attribution is *verified rather than inferred*. Instead of reading the type
assertions on the event and guessing which rule produced them, each of the eleven
rule bodies is executed against the graph and checked for whether it yields
triples about this event. That distinguishes rules sharing a consequent (rules 3
and 4 both assert ValidatedEvent) and produces attribution the reader can
re-derive.

Default target is the richest event in fold 0 - one carrying ValidatedEvent,
CriticalAnomaly and BehavioralAnomaly simultaneously, so a single window
exercises validation, AAL alerting and the confidence gate at once.

Usage:
  PYTHONPATH=. python evaluation/run_worked_example.py --config config/casas_aruba.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from evaluation.run_ablation_diagnostics import _local, _rebuild_folds
from neurosymbolic_iot.reasoning_feedback.reasoning.rule_executor import (
    PREFIXES,
    RULES,
    _ensure_room_self_types,
)

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")

RULE_CATEGORY = {
    1: "sensor grounding", 2: "sensor grounding",
    3: "validation", 4: "validation",
    5: "AAL anomaly", 6: "AAL anomaly", 7: "AAL anomaly",
    8: "feedback trigger", 9: "feedback trigger",
    10: "feedback trigger", 11: "feedback trigger",
}


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
# Stage 1 - raw sensor events
# ---------------------------------------------------------------------------

def rebuild_fold_sequences(cfg: Dict[str, Any], fold: int, k: int, seed: int):
    """Rebuild the held-out sequences for one fold, preserving prediction order."""
    from sklearn.model_selection import StratifiedKFold

    from neurosymbolic_iot.cli.train_neural import _ensure_utc_tz
    from neurosymbolic_iot.neural_perception.casas_sequence import (
        build_casas_sequences,
        build_casas_windows_from_raw,
    )

    np_cfg = cfg.get("neural_perception", {}).get("casas", {})
    ds_cfg = cfg.get("datasets", {}).get("casas", {})
    max_seq_len = int(np_cfg.get("max_seq_len", 256))

    dfw = build_casas_windows_from_raw(
        cfg,
        window_minutes=int(np_cfg.get("window_minutes", ds_cfg.get("window_minutes", 30))),
        stride_minutes=int(np_cfg.get("stride_minutes", ds_cfg.get("stride_minutes", 5))),
        min_events=int(np_cfg.get("min_events", ds_cfg.get("min_events_per_window", 1))),
    )
    for col in ("start_time", "end_time"):
        if col in dfw.columns:
            dfw = _ensure_utc_tz(dfw, col)

    _, full_vocab = build_casas_sequences(cfg, dfw, max_seq_len=max_seq_len, vocab=None)

    labels = dfw["label"].astype(str).values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(skf.split(dfw, labels))
    _, val_idx = splits[fold]

    df_val = dfw.iloc[val_idx].reset_index(drop=True)
    for col in ("start_time", "end_time"):
        if col in df_val.columns:
            df_val = _ensure_utc_tz(df_val, col)
    val_seqs, _ = build_casas_sequences(cfg, df_val, max_seq_len=max_seq_len, vocab=full_vocab)
    return val_seqs, full_vocab


def raw_events_for_window(window, vocab: Dict[str, int]) -> Dict[str, Any]:
    """The exact sensor event sequence the network consumed, with timestamps."""
    id2tok = {int(idx): str(tok) for tok, idx in vocab.items()}
    events: List[Dict[str, Any]] = []
    for token_id, delta in zip(window.token_ids, window.time_deltas):
        token = id2tok.get(int(token_id), "<UNK>")
        if token in ("<PAD>",):
            continue
        sensor, _, state = token.partition(":")
        events.append({
            "offset_seconds": round(float(delta), 1),
            "timestamp": (window.start_time + timedelta(seconds=float(delta))).isoformat(),
            "token": token,
            "sensor_id": sensor,
            "state": state or None,
        })
    return {
        "window_start": window.start_time.isoformat(),
        "window_end": window.end_time.isoformat(),
        "ground_truth_label": str(window.label),
        "n_events": len(events),
        "events": events,
    }


# ---------------------------------------------------------------------------
# Stage 2/3 - prediction and emitted triples
# ---------------------------------------------------------------------------

def _triples_about(g: Graph, subject: URIRef) -> List[Tuple[str, str, str]]:
    return sorted(
        (_local(s), _local(p), _local(o) if isinstance(o, URIRef) else str(o))
        for s, p, o in g.triples((subject, None, None))
    )


def prediction_and_triples(g: Graph, idx: int) -> Dict[str, Any]:
    pred_uri = NSIOT[f"pred_{idx}"]
    alt_uri = NSIOT[f"pred_{idx}_alt"]
    event_uri = NSIOT[f"event_{idx}"]
    time_uri = NSIOT[f"time_{idx}"]

    def _one(subject: URIRef, prop: URIRef) -> Optional[str]:
        for _, _, o in g.triples((subject, prop, None)):
            return _local(o) if isinstance(o, URIRef) else str(o)
        return None

    conf = _one(pred_uri, NSIOT["hasConfidenceScore"])
    alt_conf = _one(alt_uri, NSIOT["hasConfidenceScore"])
    triples: List[Tuple[str, str, str]] = []
    for subject in (pred_uri, alt_uri, event_uri, time_uri, NSIOT[f"nightctx_{idx}"]):
        triples.extend(_triples_about(g, subject))

    return {
        "top1_label": _one(pred_uri, NSIOT["predictsActivity"]),
        "top1_confidence": float(conf) if conf is not None else None,
        "top2_label": _one(alt_uri, NSIOT["predictsActivity"]),
        "top2_confidence": float(alt_conf) if alt_conf is not None else None,
        "margin": (
            round(float(conf) - float(alt_conf), 4)
            if conf is not None and alt_conf is not None else None
        ),
        "model_tag": _one(pred_uri, NSIOT["generatedByModel"]),
        "generated_at": _one(pred_uri, NSIOT["generatedAt"]),
        "emitted_triples": [list(t) for t in triples],
        "n_emitted_triples": len(triples),
    }


# ---------------------------------------------------------------------------
# Stage 4 - verified rule attribution
# ---------------------------------------------------------------------------

def attribute_rules(g: Graph, idx: int) -> List[Dict[str, Any]]:
    """Fire each rule and record whether it yields triples about this event.

    Executing the bodies rather than reading the event's types is what makes the
    attribution checkable, and it separates rules that share a consequent.
    """
    event_uri = NSIOT[f"event_{idx}"]
    person_uri = NSIOT["Participant1"]
    out: List[Dict[str, Any]] = []

    for rule_id, label, query in RULES:
        try:
            produced = list(g.query(PREFIXES + query))
        except Exception as exc:
            out.append({"rule_id": rule_id, "label": label, "fired_on_this_event": False,
                        "error": str(exc)})
            continue

        about_event: List[List[str]] = []
        about_person: List[List[str]] = []
        for triple in produced:
            if len(triple) != 3:
                continue
            s, p, o = triple
            rendered = [_local(s), _local(p), _local(o) if isinstance(o, URIRef) else str(o)]
            if s == event_uri:
                about_event.append(rendered)
            elif s == person_uri:
                about_person.append(rendered)

        out.append({
            "rule_id": rule_id,
            "category": RULE_CATEGORY.get(rule_id, "unknown"),
            "label": label,
            "fired_on_this_event": bool(about_event),
            "asserted_about_event": about_event,
            "asserted_about_person": about_person if about_event else [],
            "total_matches_graph_wide": len(produced),
        })
    return out


def supporting_evidence(g: Graph, idx: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    """The graph facts that make the fired rules true — the explanation trace."""
    sensors_in_window = sorted({e["sensor_id"] for e in raw["events"]})
    sensor_facts: List[List[str]] = []
    for sensor_id in sensors_in_window:
        sensor_uri = NSIOT[f"sensor_{sensor_id}"]
        sensor_facts.extend([list(t) for t in _triples_about(g, sensor_uri)])

    person_facts = [list(t) for t in _triples_about(g, NSIOT["Participant1"])]
    return {
        "sensors_in_window": sensors_in_window,
        "sensor_facts_in_kg": sensor_facts,
        "person_facts_in_kg": person_facts,
        "provenance_note": (
            "The KG asserts sensor state and location globally rather than linking sensors to the "
            "event that observed them: there is no event->sensor edge. The sensor facts above are "
            "therefore matched by sensor id from the window's raw tokens, not followed from the "
            "event node. This is the same gap that forced E1 to add explicit event->room "
            "provenance triples before cross-source location conflicts could be scoped to events."
        ),
    }


# ---------------------------------------------------------------------------
# Stage 5 - final decision
# ---------------------------------------------------------------------------

def final_decision(
    pred: Dict[str, Any],
    raw: Dict[str, Any],
    rules: List[Dict[str, Any]],
    base_threshold: float,
    raised_threshold: float,
) -> Dict[str, Any]:
    conf = pred["top1_confidence"] or 0.0
    fired = [r for r in rules if r["fired_on_this_event"]]
    feedback = [r for r in fired if r["category"] == "feedback trigger"]
    alerts = [r for r in fired if r["category"] == "AAL anomaly"]
    validations = [r for r in fired if r["category"] == "validation"]

    correct = pred["top1_label"] == raw["ground_truth_label"]
    return {
        "predicted_label": pred["top1_label"],
        "ground_truth_label": raw["ground_truth_label"],
        "prediction_correct": correct,
        "confidence": conf,
        "answered_at_base_gate": conf >= base_threshold,
        "answered_at_raised_gate": conf >= raised_threshold,
        "base_gate": base_threshold,
        "raised_gate": raised_threshold,
        "rules_fired": [r["rule_id"] for r in fired],
        "validated_by": [r["rule_id"] for r in validations],
        "alerts_raised": [
            {"rule_id": r["rule_id"], "assertion": r["asserted_about_event"]} for r in alerts
        ],
        "feedback_flags": [
            {"rule_id": r["rule_id"], "assertion": r["asserted_about_event"]} for r in feedback
        ],
        "label_changed_by_symbolic_layer": False,
        "label_change_note": (
            "The symbolic layer never rewrites the argmax label (E2): validation annotates, and "
            "feedback flags are applied only to predictions already below the confidence gate. "
            "The final label is the network's top-1."
        ),
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    raw = payload["stage_1_raw_sensor_events"]
    pred = payload["stage_2_neural_prediction"]
    rules = payload["stage_4_rule_firings"]
    dec = payload["stage_5_final_decision"]
    ev = payload["stage_4b_supporting_evidence"]
    meta = payload["_meta"]

    lines: List[str] = []
    lines.append("# Worked end-to-end example (E10 — reviewer item R2-8)")
    lines.append("")
    lines.append(f"A single real window from the published Aruba run: fold {meta['fold']}, "
                 f"event index {meta['event_index']} (`{meta['event_uri']}`).")
    lines.append("")

    lines.append("## Stage 1 — Raw sensor events")
    lines.append("")
    lines.append(f"Window `{raw['window_start']}` → `{raw['window_end']}` "
                 f"({raw['n_events']} events). Annotated activity: **{raw['ground_truth_label']}**.")
    lines.append("")
    lines.append("| t+s | Sensor | State |")
    lines.append("|---|---|---|")
    shown = raw["events"][: meta["max_events_shown"]]
    for e in shown:
        lines.append(f"| {e['offset_seconds']:.0f} | `{e['sensor_id']}` | {e['state'] or '—'} |")
    if len(raw["events"]) > len(shown):
        lines.append(f"| … | *({len(raw['events']) - len(shown)} further events)* | |")
    lines.append("")

    lines.append("## Stage 2 — Neural perception")
    lines.append("")
    lines.append(f"- Model: `{pred['model_tag']}` (CASAS GRU)")
    lines.append(f"- Top-1: **{pred['top1_label']}** at confidence **{pred['top1_confidence']:.4f}**")
    if pred["top2_label"]:
        lines.append(f"- Top-2: {pred['top2_label']} at {pred['top2_confidence']:.4f} "
                     f"(margin {pred['margin']:.4f})")
    lines.append("")

    lines.append("## Stage 3 — Triples emitted into the knowledge graph")
    lines.append("")
    lines.append(f"{pred['n_emitted_triples']} triples describe this window:")
    lines.append("")
    lines.append("```turtle")
    for s, p, o in pred["emitted_triples"][: meta["max_triples_shown"]]:
        obj = o if o.startswith(("http", "2012", "2013")) or o[0].isdigit() else o
        lines.append(f"nsiot:{s}  nsiot:{p}  {obj} .")
    if pred["n_emitted_triples"] > meta["max_triples_shown"]:
        lines.append(f"# … {pred['n_emitted_triples'] - meta['max_triples_shown']} further triples")
    lines.append("```")
    lines.append("")

    lines.append("## Stage 4 — Rule firings")
    lines.append("")
    lines.append("Each of the eleven rule bodies was executed against the graph; a rule is listed as")
    lines.append("fired only if it yields a triple whose subject is this event.")
    lines.append("")
    lines.append("| Rule | Category | Fired here? | Asserted |")
    lines.append("|---|---|---|---|")
    for r in rules:
        assertion = "; ".join(f"`{p}` → `{o}`" for _, p, o in r.get("asserted_about_event", [])) or "—"
        lines.append(f"| R{r['rule_id']} | {r['category']} | "
                     f"{'**yes**' if r['fired_on_this_event'] else 'no'} | {assertion} |")
    lines.append("")

    lines.append("### Supporting evidence (why those rules matched)")
    lines.append("")
    lines.append(f"Sensors active in this window: {', '.join('`' + s + '`' for s in ev['sensors_in_window'])}")
    lines.append("")
    for s, p, o in ev["person_facts_in_kg"]:
        lines.append(f"- `{s}` `{p}` `{o}`")
    lines.append("")
    lines.append(f"*{ev['provenance_note']}*")
    lines.append("")

    lines.append("## Stage 5 — Final decision and explanation trace")
    lines.append("")
    lines.append(f"- **Decision:** {dec['predicted_label']} "
                 f"(ground truth {dec['ground_truth_label']} — "
                 f"{'correct' if dec['prediction_correct'] else 'INCORRECT'})")
    lines.append(f"- **Confidence** {dec['confidence']:.4f}; answered at the {dec['base_gate']} gate: "
                 f"{'yes' if dec['answered_at_base_gate'] else 'no'}; at the post-feedback "
                 f"{dec['raised_gate']} gate: {'yes' if dec['answered_at_raised_gate'] else 'no'}")
    lines.append(f"- **Validated by:** {dec['validated_by'] or 'none'}")
    lines.append(f"- **Alerts raised:** "
                 + (", ".join(f"R{a['rule_id']}" for a in dec["alerts_raised"]) or "none"))
    lines.append(f"- **Feedback flags:** "
                 + (", ".join(f"R{f['rule_id']}" for f in dec["feedback_flags"]) or "none"))
    lines.append(f"- **Label changed by the symbolic layer:** no")
    lines.append("")
    lines.append(f"*{dec['label_change_note']}*")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E10: worked end-to-end example from a real window.")
    p.add_argument("--config", default="config/casas_aruba.yaml")
    p.add_argument("--results-dir", default="evaluation/results_aruba")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--event-index", type=int, default=192,
                   help="Prediction/event index within the fold (default: the richest event)")
    p.add_argument("--base-threshold", type=float, default=0.70)
    p.add_argument("--raised-threshold", type=float, default=0.75)
    p.add_argument("--max-events-shown", type=int, default=25)
    p.add_argument("--max-triples-shown", type=int, default=30)
    p.add_argument("--outdir", default="evaluation/results_example")
    return p.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    fold_dir = Path(args.results_dir) / f"fold_{args.fold}"
    kg_path = fold_dir / "kg" / "casas" / "populated_kg.ttl"
    if not kg_path.exists():
        log.error("KG not found: %s", kg_path)
        return 1

    log.info("Rebuilding fold %d sequences ...", args.fold)
    val_seqs, vocab = rebuild_fold_sequences(cfg, args.fold, args.k, args.seed)
    if args.event_index >= len(val_seqs):
        log.error("event index %d out of range (fold has %d windows)", args.event_index, len(val_seqs))
        return 1

    log.info("Loading fold KG ...")
    g = Graph()
    g.parse(str(kg_path), format="turtle")
    _ensure_room_self_types(g)

    window = val_seqs[args.event_index]
    raw = raw_events_for_window(window, vocab)
    pred = prediction_and_triples(g, args.event_index)
    rules = attribute_rules(g, args.event_index)
    evidence = supporting_evidence(g, args.event_index, raw)
    decision = final_decision(pred, raw, rules, args.base_threshold, args.raised_threshold)

    payload = {
        "_meta": {
            "script": "evaluation/run_worked_example.py",
            "commit": _git_commit(),
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source_run": "evaluation/results_aruba (commit f5c6cd3)",
            "config": args.config,
            "fold": args.fold,
            "event_index": args.event_index,
            "event_uri": f"nsiot:event_{args.event_index}",
            "max_events_shown": args.max_events_shown,
            "max_triples_shown": args.max_triples_shown,
            "attribution_method": (
                "Each rule body is executed against the stored graph and counted as fired only if "
                "it yields a triple whose subject is this event. Reading the event's type "
                "assertions instead would not separate rules sharing a consequent (3 and 4 both "
                "assert ValidatedEvent) and would not be re-derivable by a reader."
            ),
        },
        "stage_1_raw_sensor_events": raw,
        "stage_2_neural_prediction": pred,
        "stage_3_emitted_triples_count": pred["n_emitted_triples"],
        "stage_4_rule_firings": rules,
        "stage_4b_supporting_evidence": evidence,
        "stage_5_final_decision": decision,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"worked_example_fold{args.fold}_event{args.event_index}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    fired = [r for r in rules if r["fired_on_this_event"]]
    print()
    print("=" * 92)
    print(f"  E10 — WORKED EXAMPLE (fold {args.fold}, event {args.event_index})")
    print("=" * 92)
    print(f"  Stage 1  raw sensor events : {raw['n_events']} events, "
          f"{raw['window_start']} -> {raw['window_end']}")
    print(f"           annotated activity: {raw['ground_truth_label']}")
    print(f"  Stage 2  neural prediction : {pred['top1_label']} @ {pred['top1_confidence']:.4f}"
          + (f"  (top-2 {pred['top2_label']} @ {pred['top2_confidence']:.4f}, margin {pred['margin']:.4f})"
             if pred["top2_label"] else ""))
    print(f"  Stage 3  triples emitted   : {pred['n_emitted_triples']}")
    print(f"  Stage 4  rules fired       : {[r['rule_id'] for r in fired]}")
    for r in fired:
        for _, p, o in r["asserted_about_event"]:
            print(f"             R{r['rule_id']:<3} ({r['category']:<16}) -> {p} {o}")
    print(f"  Stage 5  decision          : {decision['predicted_label']} "
          f"(truth {decision['ground_truth_label']}, "
          f"{'correct' if decision['prediction_correct'] else 'INCORRECT'})")
    print(f"           answered @{decision['base_gate']}: {decision['answered_at_base_gate']}, "
          f"@{decision['raised_gate']}: {decision['answered_at_raised_gate']}")
    print(f"           label changed by symbolic layer: no")
    print("=" * 92)

    log.info("Wrote %s and %s", out_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
