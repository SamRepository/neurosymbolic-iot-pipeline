"""
Over-correction / false-retraction safety metrics (E4 - FGCS revision)
=====================================================================
Addresses reviewer item R1-10: the error analysis identifies over-correction of
valid out-of-distribution behaviour as a primary failure mode, but no metric
quantifies it. This bears directly on the healthcare / AAL safety claim, so the
rate belongs next to the false-positive-reduction figure.

Reuses the validated reconstruction from ``run_ablation_diagnostics`` (per-window
predictions rebuilt from each fold's serialized KG, ground truth from a
deterministic window/fold rebuild) and therefore inherits its hard validation
gate against the published AI-Only metrics.

Four distinct over-correction pathways are measured separately, because they have
different operational consequences and only one of them is what the request
anticipated:

1. **Flagging (rules 8-11).** A FeedbackRequired flag raised on a window the
   network classified *correctly*. Operationally this is a needless review
   request - the symbolic layer questioned a prediction that was right.
2. **Retraction (rule 8 path).** ``adapt_kg.retract_false_positives`` deletes
   triples only for ``FalsePositiveHallucination`` flags, which only rule 8
   emits. Reported separately because a rate of zero here can mean "safe" or
   "never activated", and those must not be conflated.
3. **Abstention.** The feedback cycle raises the confidence gate (E2), so windows
   between the old and new gate stop being answered at all. Correct predictions
   in that band are valid behaviours the system silently stopped reporting - the
   closest analogue to "valid behaviour retracted" in the deployed pipeline.
4. **AAL alerting (rules 5-7).** CriticalAnomaly / BehavioralAnomaly alerts are
   what would actually reach a caregiver. An alert raised on a *misclassified*
   window rests on a false premise. Note the ceiling on what is knowable here:
   CASAS Aruba carries activity labels, not anomaly labels, so an alert on a
   correctly classified window cannot be adjudicated true or false from the data.

Outputs ``evaluation/results_aruba/safety_metrics.json`` and a markdown summary.

Usage:
  PYTHONPATH=. python evaluation/run_safety_metrics.py --config config/casas_aruba.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.run_ablation_diagnostics import (
    OVERRIDE_ERROR_TYPES,
    RULE_CATEGORIES,
    Prediction,
    _event_index,
    _git_commit,
    _load_fold_predictions,
    _load_reasoning,
    _local,
    _rebuild_folds,
)

log = logging.getLogger(__name__)

# Which rule emits which FeedbackRequired error type (rule_executor.RULES 8-11).
ERROR_TYPE_TO_RULE: Dict[str, int] = {
    "FalsePositiveHallucination": 8,
    "ContextualMismatch": 9,
    "MutuallyExclusiveActivities": 10,
    "UnsupportedClaim": 11,
}

# Which rule emits which AAL alert type (rule_executor.RULES 5-7).
ALERT_TYPE_TO_RULE: Dict[str, int] = {
    "UnattendedAppliance": 5,
    "UnattendedFireHazard": 6,
    "NocturnalActivity": 7,
}


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def analyse_fold(
    fold: int,
    fold_dir: Path,
    ground_truth: List[str],
    base_threshold: float,
    raised_threshold: float,
) -> Dict[str, Any]:
    preds = _load_fold_predictions(fold_dir / "kg" / "casas" / "populated_kg.ttl", ground_truth)
    by_index: Dict[int, Prediction] = {p.index: p for p in preds}
    rr = _load_reasoning(fold_dir)

    # ---- 1. Flagging over-correction (rules 8-11) ----
    flag_rows: List[Tuple[int, int, bool]] = []  # (event_index, rule_id, prediction_was_correct)
    for flag in rr["feedback_flags"]:
        idx = _event_index(flag.get("uri", ""))
        err = _local(flag.get("error_type", ""))
        if idx is None or idx not in by_index:
            continue
        rule_id = ERROR_TYPE_TO_RULE.get(err)
        if rule_id is None:
            continue
        p = by_index[idx]
        flag_rows.append((idx, rule_id, p.label == p.ground_truth))

    by_rule: Dict[int, Dict[str, int]] = {}
    for _, rule_id, was_correct in flag_rows:
        entry = by_rule.setdefault(rule_id, {"n_flags": 0, "n_on_correct": 0})
        entry["n_flags"] += 1
        entry["n_on_correct"] += int(was_correct)
    for entry in by_rule.values():
        entry["false_flag_rate"] = _rate(entry["n_on_correct"], entry["n_flags"])

    n_flags = len(flag_rows)
    n_flags_on_correct = sum(1 for _, _, ok in flag_rows if ok)

    # ---- 2. Retraction (rule 8 path only) ----
    n_retraction_eligible = sum(
        1 for flag in rr["feedback_flags"]
        if "FalsePositiveHallucination" in flag.get("error_type", "")
    )
    retraction_eligible_on_correct = sum(
        1 for idx, rule_id, ok in flag_rows if rule_id == 8 and ok
    )

    # ---- 3. Abstention over-correction (the gate rise) ----
    abstained = [
        p for p in preds
        if base_threshold <= p.confidence < raised_threshold
    ]
    n_abstained_correct = sum(1 for p in abstained if p.label == p.ground_truth)
    active_at_base = [p for p in preds if p.confidence >= base_threshold]
    n_correct_at_base = sum(1 for p in active_at_base if p.label == p.ground_truth)

    # ---- 4. AAL alerting (rules 5-7) ----
    alert_rows: List[Tuple[int, int, bool]] = []
    seen_alert: set = set()
    for key in ("critical_anomalies", "behavioral_anomalies"):
        for alert in rr.get(key, []):
            idx = _event_index(alert.get("uri", ""))
            alert_type = _local(alert.get("alert_type", ""))
            rule_id = ALERT_TYPE_TO_RULE.get(alert_type)
            if idx is None or rule_id is None or idx not in by_index:
                continue
            dedup = (idx, rule_id)
            if dedup in seen_alert:
                continue
            seen_alert.add(dedup)
            p = by_index[idx]
            alert_rows.append((idx, rule_id, p.label == p.ground_truth))

    alerts_by_rule: Dict[int, Dict[str, int]] = {}
    for _, rule_id, was_correct in alert_rows:
        entry = alerts_by_rule.setdefault(rule_id, {"n_alerts": 0, "n_on_misclassified": 0})
        entry["n_alerts"] += 1
        entry["n_on_misclassified"] += int(not was_correct)
    for entry in alerts_by_rule.values():
        entry["false_premise_rate"] = _rate(entry["n_on_misclassified"], entry["n_alerts"])

    n_alerts = len(alert_rows)
    n_alerts_false_premise = sum(1 for _, _, ok in alert_rows if not ok)

    return {
        "fold": fold,
        "n_predictions": len(preds),
        "flagging_rules_8_11": {
            "n_flags": n_flags,
            "n_flags_on_correct_predictions": n_flags_on_correct,
            "false_flag_rate": _rate(n_flags_on_correct, n_flags),
            "by_rule": {str(k): v for k, v in sorted(by_rule.items())},
        },
        "retraction_rule_8": {
            "n_retraction_eligible_flags": n_retraction_eligible,
            "n_eligible_on_correct_predictions": retraction_eligible_on_correct,
            "false_retraction_rate": _rate(retraction_eligible_on_correct, n_retraction_eligible),
            "triples_retracted": 0 if n_retraction_eligible == 0 else None,
        },
        "abstention": {
            "n_active_at_base_gate": len(active_at_base),
            "n_abstained_by_gate_rise": len(abstained),
            "n_abstained_that_were_correct": n_abstained_correct,
            "over_correction_rate_within_abstained": _rate(n_abstained_correct, len(abstained)),
            "correct_predictions_lost_as_pct_of_correct_at_base": _rate(
                n_abstained_correct, n_correct_at_base
            ),
        },
        "aal_alerts_rules_5_7": {
            "n_alerts": n_alerts,
            "n_alerts_on_misclassified_windows": n_alerts_false_premise,
            "false_premise_rate": _rate(n_alerts_false_premise, n_alerts),
            "by_rule": {str(k): v for k, v in sorted(alerts_by_rule.items())},
        },
    }


def _pool(fold_results: List[Dict[str, Any]], path: List[str], key: str) -> int:
    total = 0
    for fr in fold_results:
        node: Any = fr
        for step in path:
            node = node[step]
        total += int(node.get(key, 0) or 0)
    return total


def build_summary(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    flags = _pool(fold_results, ["flagging_rules_8_11"], "n_flags")
    flags_correct = _pool(fold_results, ["flagging_rules_8_11"], "n_flags_on_correct_predictions")
    eligible = _pool(fold_results, ["retraction_rule_8"], "n_retraction_eligible_flags")
    abstained = _pool(fold_results, ["abstention"], "n_abstained_by_gate_rise")
    abstained_correct = _pool(fold_results, ["abstention"], "n_abstained_that_were_correct")
    active_base = _pool(fold_results, ["abstention"], "n_active_at_base_gate")
    alerts = _pool(fold_results, ["aal_alerts_rules_5_7"], "n_alerts")
    alerts_false = _pool(fold_results, ["aal_alerts_rules_5_7"], "n_alerts_on_misclassified_windows")

    rule_flag_totals: Dict[str, Dict[str, Any]] = {}
    for fr in fold_results:
        for rid, entry in fr["flagging_rules_8_11"]["by_rule"].items():
            acc = rule_flag_totals.setdefault(rid, {"n_flags": 0, "n_on_correct": 0})
            acc["n_flags"] += entry["n_flags"]
            acc["n_on_correct"] += entry["n_on_correct"]
    for rid, acc in rule_flag_totals.items():
        acc["false_flag_rate"] = _rate(acc["n_on_correct"], acc["n_flags"])
        acc["category"] = RULE_CATEGORIES[int(rid)]

    rule_alert_totals: Dict[str, Dict[str, Any]] = {}
    for fr in fold_results:
        for rid, entry in fr["aal_alerts_rules_5_7"]["by_rule"].items():
            acc = rule_alert_totals.setdefault(rid, {"n_alerts": 0, "n_on_misclassified": 0})
            acc["n_alerts"] += entry["n_alerts"]
            acc["n_on_misclassified"] += entry["n_on_misclassified"]
    for rid, acc in rule_alert_totals.items():
        acc["false_premise_rate"] = _rate(acc["n_on_misclassified"], acc["n_alerts"])
        acc["category"] = RULE_CATEGORIES[int(rid)]

    per_fold_rates = [
        fr["flagging_rules_8_11"]["false_flag_rate"] for fr in fold_results
        if fr["flagging_rules_8_11"]["false_flag_rate"] is not None
    ]

    return {
        "headline_false_flag_rate": _rate(flags_correct, flags),
        "flagging_rules_8_11": {
            "n_flags_total": flags,
            "n_on_correct_predictions": flags_correct,
            "false_flag_rate_pooled": _rate(flags_correct, flags),
            "false_flag_rate_per_fold": per_fold_rates,
            "false_flag_rate_mean_of_folds": round(float(np.mean(per_fold_rates)), 4) if per_fold_rates else None,
            "by_rule": rule_flag_totals,
        },
        "retraction_rule_8": {
            "n_retraction_eligible_flags_total": eligible,
            "triples_retracted_total": 0 if eligible == 0 else None,
            "status": (
                "MECHANISM NEVER ACTIVATED" if eligible == 0
                else "activated - see per-fold detail"
            ),
        },
        "abstention": {
            "n_active_at_base_gate_total": active_base,
            "n_abstained_total": abstained,
            "n_abstained_that_were_correct": abstained_correct,
            "over_correction_rate_within_abstained": _rate(abstained_correct, abstained),
            "correct_predictions_dropped_as_pct_of_active_set": _rate(abstained_correct, active_base),
        },
        "aal_alerts_rules_5_7": {
            "n_alerts_total": alerts,
            "n_on_misclassified_windows": alerts_false,
            "false_premise_rate_pooled": _rate(alerts_false, alerts),
            "by_rule": rule_alert_totals,
        },
    }


def build_findings(summary: Dict[str, Any]) -> Dict[str, str]:
    fl = summary["flagging_rules_8_11"]
    ab = summary["abstention"]
    al = summary["aal_alerts_rules_5_7"]
    rt = summary["retraction_rule_8"]

    return {
        "headline": (
            f"{fl['n_on_correct_predictions']} of {fl['n_flags_total']} feedback flags "
            f"({fl['false_flag_rate_pooled']*100:.1f} %) were raised on windows the network had "
            "classified correctly. This is the false-flag rate the manuscript should report "
            "beside the false-positive-reduction figure."
        ),
        "retraction": (
            "No triples were retracted anywhere in the run. adapt_kg.retract_false_positives acts "
            "only on FalsePositiveHallucination flags, which only rule 8 emits, and rule 8 never "
            f"fired on Aruba ({rt['n_retraction_eligible_flags_total']} eligible flags). The "
            "false-retraction rate is therefore undefined rather than zero, and the manuscript "
            "must not report it as evidence that retraction is safe: the mechanism was never "
            "exercised on this dataset."
        ),
        "abstention": (
            f"The over-correction that did occur is abstention. Raising the gate dropped "
            f"{ab['n_abstained_total']} windows across the five folds, of which "
            f"{ab['n_abstained_that_were_correct']} "
            f"({ab['over_correction_rate_within_abstained']*100:.1f} %) carried correct "
            "predictions. Those are valid behaviours the system silently stopped reporting - the "
            "closest analogue in this pipeline to the reviewer's 'valid atypical behaviour "
            "erroneously retracted', and the number that matters for an AAL deployment, since an "
            "unanswered window raises no alert at all."
        ),
        "aal_alerts": (
            f"Of {al['n_alerts_total']} AAL alerts raised by rules 5-7, "
            f"{al['n_on_misclassified_windows']} ({al['false_premise_rate_pooled']*100:.1f} %) were "
            "raised on windows whose underlying activity classification was wrong, so the alert "
            "rests on a false premise. This is a lower bound on the alert false-alarm rate, not "
            "the rate itself."
        ),
        "measurement_ceiling": (
            "CASAS Aruba provides activity labels, not anomaly labels. An alert raised on a "
            "correctly classified window therefore cannot be adjudicated as a true or false "
            "anomaly from the data - only alerts built on a misclassification can be identified "
            "with certainty. Any fuller safety claim needs a dataset with annotated anomalies or "
            "expert adjudication, and the manuscript should say so rather than implying the "
            "false-alarm rate is fully characterised."
        ),
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    s = payload["summary"]
    f = payload["findings"]
    lines: List[str] = []
    lines.append("# Over-correction and false-retraction safety metrics (E4 — reviewer item R1-10)")
    lines.append("")
    lines.append(f"Source: {payload['_meta']['analysed_run']}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"{f['headline']}")
    lines.append("")
    lines.append("| Pathway | Denominator | Over-corrections | Rate |")
    lines.append("|---|---|---|---|")
    fl = s["flagging_rules_8_11"]
    ab = s["abstention"]
    al = s["aal_alerts_rules_5_7"]
    lines.append(
        f"| Flagged though correct (rules 8–11) | {fl['n_flags_total']} flags "
        f"| {fl['n_on_correct_predictions']} | **{fl['false_flag_rate_pooled']*100:.1f} %** |"
    )
    lines.append(
        f"| Retracted though correct (rule 8) | {s['retraction_rule_8']['n_retraction_eligible_flags_total']} eligible "
        f"| — | undefined — mechanism never activated |"
    )
    lines.append(
        f"| Correct predictions abstained on | {ab['n_abstained_total']} abstained "
        f"| {ab['n_abstained_that_were_correct']} | **{ab['over_correction_rate_within_abstained']*100:.1f} %** |"
    )
    lines.append(
        f"| AAL alerts on a false premise (rules 5–7) | {al['n_alerts_total']} alerts "
        f"| {al['n_on_misclassified_windows']} | **{al['false_premise_rate_pooled']*100:.1f} %** |"
    )
    lines.append("")
    lines.append("## By rule")
    lines.append("")
    lines.append("| Rule | Category | Events | Over-corrections | Rate |")
    lines.append("|---|---|---|---|---|")
    for rid, acc in sorted(fl["by_rule"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| R{rid} | {acc['category']} | {acc['n_flags']} flags | {acc['n_on_correct']} on correct "
            f"| {acc['false_flag_rate']*100:.1f} % |"
        )
    for rid, acc in sorted(al["by_rule"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| R{rid} | {acc['category']} | {acc['n_alerts']} alerts | {acc['n_on_misclassified']} false premise "
            f"| {acc['false_premise_rate']*100:.1f} % |"
        )
    lines.append("")
    for heading, key in (
        ("Retraction", "retraction"),
        ("Abstention", "abstention"),
        ("AAL alerts", "aal_alerts"),
        ("What this data cannot establish", "measurement_ceiling"),
    ):
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(f[key])
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4: over-correction / false-retraction safety metrics.")
    p.add_argument("--config", default="config/casas_aruba.yaml")
    p.add_argument("--results-dir", default="evaluation/results_aruba")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-threshold", type=float, default=0.70)
    p.add_argument("--raised-threshold", type=float, default=0.75)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    results_dir = Path(args.results_dir)
    log.info("Rebuilding window table and %d-fold split (seed %d) ...", args.k, args.seed)
    fold_ground_truth = _rebuild_folds(cfg, args.k, args.seed)

    fold_results: List[Dict[str, Any]] = []
    for fold in range(args.k):
        fr = analyse_fold(
            fold, results_dir / f"fold_{fold}", fold_ground_truth[fold],
            args.base_threshold, args.raised_threshold,
        )
        fold_results.append(fr)
        log.info(
            "fold %d: flags %d (%d on correct), abstained %d (%d correct), alerts %d (%d false premise)",
            fold,
            fr["flagging_rules_8_11"]["n_flags"],
            fr["flagging_rules_8_11"]["n_flags_on_correct_predictions"],
            fr["abstention"]["n_abstained_by_gate_rise"],
            fr["abstention"]["n_abstained_that_were_correct"],
            fr["aal_alerts_rules_5_7"]["n_alerts"],
            fr["aal_alerts_rules_5_7"]["n_alerts_on_misclassified_windows"],
        )

    summary = build_summary(fold_results)
    findings = build_findings(summary)

    payload = {
        "_meta": {
            "script": "evaluation/run_safety_metrics.py",
            "commit": _git_commit(),
            "generated": datetime.now().isoformat(timespec="seconds"),
            "analysed_run": "evaluation/results_aruba/ablation_cv.json (commit f5c6cd3)",
            "config": args.config,
            "thresholds": {"base": args.base_threshold, "raised": args.raised_threshold},
            "method": (
                "Reuses the reconstruction validated in run_ablation_diagnostics.py, which "
                "reproduces the published per-fold AI-Only metrics to 1e-9."
            ),
            "definitions": {
                "false_flag": "a FeedbackRequired flag (rules 8-11) on a correctly classified window",
                "false_retraction": "a triple deletion (rule 8 path) affecting a correctly classified window",
                "abstention_over_correction": (
                    "a correct prediction that stopped being answered when the feedback cycle "
                    "raised the confidence gate"
                ),
                "false_premise_alert": (
                    "an AAL alert (rules 5-7) raised on a window whose activity classification was wrong"
                ),
            },
        },
        "summary": summary,
        "findings": findings,
        "per_fold": fold_results,
    }

    out_path = Path(args.out) if args.out else results_dir / "safety_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    fl = summary["flagging_rules_8_11"]
    ab = summary["abstention"]
    al = summary["aal_alerts_rules_5_7"]
    print()
    print("=" * 92)
    print("  E4 — OVER-CORRECTION / FALSE-RETRACTION SAFETY METRICS")
    print("=" * 92)
    print(f"  Flagged though correct (rules 8-11) : {fl['n_on_correct_predictions']}/{fl['n_flags_total']}"
          f"  = {fl['false_flag_rate_pooled']*100:.1f} %   <-- headline false-flag rate")
    print(f"  Retraction (rule 8 path)            : {summary['retraction_rule_8']['status']}"
          f"  ({summary['retraction_rule_8']['n_retraction_eligible_flags_total']} eligible flags)")
    print(f"  Correct predictions abstained on    : {ab['n_abstained_that_were_correct']}/{ab['n_abstained_total']}"
          f"  = {ab['over_correction_rate_within_abstained']*100:.1f} %"
          f"  ({ab['correct_predictions_dropped_as_pct_of_active_set']*100:.1f} % of the active set)")
    print(f"  AAL alerts on a false premise (5-7) : {al['n_on_misclassified_windows']}/{al['n_alerts_total']}"
          f"  = {al['false_premise_rate_pooled']*100:.1f} %")
    print("  By rule:")
    for rid, acc in sorted(fl["by_rule"].items(), key=lambda kv: int(kv[0])):
        print(f"    R{rid:<3} {acc['category']:<18} {acc['n_flags']:>4} flags  "
              f"{acc['n_on_correct']:>4} on correct  ({acc['false_flag_rate']*100:.1f} %)")
    for rid, acc in sorted(al["by_rule"].items(), key=lambda kv: int(kv[0])):
        print(f"    R{rid:<3} {acc['category']:<18} {acc['n_alerts']:>4} alerts {acc['n_on_misclassified']:>4} "
              f"false premise ({acc['false_premise_rate']*100:.1f} %)")
    print("=" * 92)

    log.info("Wrote %s and %s", out_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
