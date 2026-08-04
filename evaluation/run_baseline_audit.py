"""
Baseline audit and comparability check (E11 - FGCS revision)
============================================================
Addresses reviewer item R1-8 (fidelity-validation procedure for the reimplemented
neuro-symbolic baselines) and the paper session's post-E2 extension (at what
confidence gate and on what coverage was each baseline scored, so Table 12 can be
made apples-to-apples).

**Primary finding, and the reason this is an audit rather than a scores dump: the
three reimplemented neuro-symbolic baselines do not exist in this repository.**
An exhaustive search (recorded in ``search_evidence`` below so it can be
re-executed) finds no implementation, no stored scores, and no deleted files in
git history. The only baselines present are the two classical models built by
``evaluation/common.py::build_model`` - LogisticRegression for CASAS and
RandomForest for SPHERE - which README maps to **Table 2 (§4.1)**, not Table 12.

The fidelity-validation procedure R1-8 asks for therefore cannot be documented
from code. Either those baselines were implemented outside this repository, or
the Table 12 numbers have another provenance; the paper session has to establish
which before that table can be defended. This script documents exactly what does
exist so the answer can be written truthfully.

What it does produce, all of which R1-8 and its extension need regardless:

  * every baseline score in the repository with its full protocol - split sizes,
    feature count, class count, model, hyperparameters;
  * the **coverage answer**: the classical baselines apply no confidence gate at
    all (``model.predict`` over the whole test set), so they are scored at 100 %
    coverage against the pipeline's 50.5 %;
  * a comparability check against the pipeline's own protocol, which turns out to
    differ in dataset preparation, class count and split scheme;
  * a flag on the CASAS baseline's perfect score, which is a degenerate artefact
    of having ten times more features than training samples.

Usage:
  PYTHONPATH=. python evaluation/run_baseline_audit.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


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


SEARCH_EVIDENCE = {
    "question": "Do implementations of the three reimplemented neuro-symbolic baselines exist?",
    "answer": "No. No implementation, no stored scores, and nothing in git history.",
    "searches_performed": [
        "grep -ri 'DeepProbLog|Logic Tensor|LTN|NeurASP|NS-CL|baseline|SOTA' over the repository",
        "find for any path matching *baseline*, *compar*, *sota*",
        "git log --all --diff-filter=AD --name-only for deleted or added baseline-named files",
        "enumeration of every model constructor reachable from evaluation/common.py::build_model",
        "enumeration of every metrics.json under outputs/",
    ],
    "what_was_found_instead": (
        "evaluation/common.py::build_model constructs exactly two estimators: "
        "LogisticRegression (CASAS default) and RandomForestClassifier (SPHERE default, or CASAS "
        "when --model rf is passed). README maps evaluation/run_experiments.py --mode baseline to "
        "Table 2 (paper section 4.1). Nothing in the repository targets Table 12."
    ),
    "implication_for_R1_8": (
        "The fidelity-validation procedure the reviewer asks about cannot be reconstructed from "
        "this repository, because the artefacts it would describe are not here. The paper session "
        "must establish the provenance of the Table 12 numbers - implemented elsewhere, quoted "
        "from the original papers, or otherwise - before the table can be defended. Reporting a "
        "fidelity procedure that cannot be evidenced would be worse than disclosing the gap."
    ),
}


def collect_baseline_scores(outputs_dir: Path) -> List[Dict[str, Any]]:
    """Every stored classical-baseline score, with the protocol that produced it."""
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(outputs_dir.glob("*/*/metrics.json")):
        run_id = metrics_path.parent.parent.name
        dataset = metrics_path.parent.name
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        feature_cols_path = metrics_path.parent / "feature_cols.json"
        features: List[str] = []
        if feature_cols_path.exists():
            features = json.loads(feature_cols_path.read_text(encoding="utf-8"))

        n_train = int(m.get("n_train", 0))
        n_features = int(m.get("n_features", len(features)))
        test = m.get("test", {}) or {}
        rows.append({
            "run_id": run_id,
            "dataset": dataset,
            "model": m.get("model"),
            "source": metrics_path.as_posix(),
            "paper_table": "Table 2 (section 4.1) per README",
            "protocol": {
                "data_source": (
                    "data/processed/%s_windows.parquet (preprocess CLI output)" % dataset
                ),
                "split": "single stratified train/val/test split, not cross-validation",
                "n_train": n_train,
                "n_val": int(m.get("n_val", 0)),
                "n_test": int(m.get("n_test", 0)),
                "n_features": n_features,
                "n_classes": int(m.get("n_classes", 0)),
                "classes": m.get("classes"),
                "seed": m.get("seed"),
                "confidence_gate": None,
                "coverage": 1.0,
                "coverage_note": (
                    "evaluation/common.py calls model.predict over the entire test set. No "
                    "confidence threshold is applied anywhere in the classical-baseline path, so "
                    "every test window is answered."
                ),
                "feature_sample": features[:10],
            },
            "scores": {
                "accuracy": test.get("accuracy"),
                "f1_macro": test.get("f1_macro"),
                "f1_weighted": test.get("f1_weighted"),
            },
            "hyperparameters": m.get("model_params"),
            "degeneracy_flags": _degeneracy_flags(n_features, n_train, test),
        })
    return rows


def _degeneracy_flags(n_features: int, n_train: int, test: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if n_train and n_features > n_train:
        flags.append(
            "n_features (%d) exceeds n_train (%d) by %.1fx - a linear model is guaranteed to "
            "separate the training set, so a high score is not evidence of a strong baseline"
            % (n_features, n_train, n_features / n_train)
        )
    f1w = test.get("f1_weighted")
    if f1w is not None and float(f1w) >= 0.999:
        flags.append(
            "perfect weighted F1 (%.3f) - treat as degenerate rather than as a baseline result "
            "worth quoting" % float(f1w)
        )
    n_test = test.get("n_test")
    return flags


def pipeline_protocol(ablation_path: Path) -> Optional[Dict[str, Any]]:
    """The protocol behind the pipeline numbers, for comparison."""
    if not ablation_path.exists():
        return None
    abl = json.loads(ablation_path.read_text(encoding="utf-8"))
    md = abl.get("metadata", {})
    n_windows = md.get("n_windows")
    k = md.get("k")
    return {
        "source": ablation_path.as_posix(),
        "data_source": "windows rebuilt from raw via build_casas_windows_from_raw (experiment config)",
        "split": "%s-fold stratified cross-validation" % k,
        "total_windows": n_windows,
        "n_classes": md.get("n_classes"),
        "classes": md.get("labels"),
        "seed": md.get("seed"),
        "confidence_gate": 0.70,
        "confidence_gate_after_feedback": 0.75,
        "coverage": 0.505,
        "coverage_note": (
            "Only predictions at or above the gate are scored (E2b): coverage is 56.1 % at the "
            "0.70 gate and 50.5 % at the post-feedback 0.75 gate."
        ),
    }


def comparability_report(
    baselines: List[Dict[str, Any]],
    pipeline: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if pipeline is None:
        return {"available": False}

    casas_rows = [b for b in baselines if b["dataset"] == "casas"]
    mismatches: List[str] = []
    if casas_rows:
        b = casas_rows[0]
        p = b["protocol"]
        if p["n_classes"] != pipeline["n_classes"]:
            mismatches.append(
                "class count differs: baseline %d vs pipeline %d"
                % (p["n_classes"], pipeline["n_classes"])
            )
        total_baseline = p["n_train"] + p["n_val"] + p["n_test"]
        if pipeline.get("total_windows") and total_baseline != pipeline["total_windows"]:
            mismatches.append(
                "window count differs: baseline %d windows (from data/processed) vs pipeline %d "
                "(rebuilt from raw)" % (total_baseline, pipeline["total_windows"])
            )
        if p["split"] != pipeline["split"]:
            mismatches.append(
                "split scheme differs: baseline uses a %s, pipeline uses %s"
                % (p["split"], pipeline["split"])
            )
        if p["coverage"] != pipeline["coverage"]:
            mismatches.append(
                "coverage differs: baseline %.1f %% (no gate) vs pipeline %.1f %% (gated)"
                % (p["coverage"] * 100, pipeline["coverage"] * 100)
            )

    return {
        "available": True,
        "coverage_matched": False,
        "mismatches": mismatches,
        "verdict": (
            "The classical baselines and the pipeline results are not directly comparable. They "
            "differ in dataset preparation, class count, split scheme and - decisively for the "
            "extension question - in coverage: the baselines answer every test window while the "
            "pipeline answers about half. Any table placing these numbers side by side needs to "
            "state all four differences, or be rebuilt on a shared protocol."
        ),
        "how_to_make_it_apples_to_apples": (
            "Two options. (a) Score the baselines under the pipeline's protocol: same rebuilt "
            "windows, same 5-fold split, and either no gate for both or the same gate for both. "
            "(b) Report the pipeline at 100 % coverage alongside its gated figure, so the reader "
            "can compare like with like. Option (b) costs nothing - AI-Only at a 0.0 gate is "
            "already derivable from the validated reconstruction in E2 - and is the honest "
            "minimum if the table is kept in its current form."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E11: baseline audit and comparability check.")
    p.add_argument("--outputs-dir", default="outputs/evaluation")
    p.add_argument("--ablation", default="evaluation/results_aruba/ablation_cv.json")
    p.add_argument("--outdir", default="evaluation/results_baselines")
    return p.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neurosymbolic_iot.utils.logging import setup_logging

    args = parse_args()
    setup_logging("INFO")

    baselines = collect_baseline_scores(Path(args.outputs_dir))
    pipeline = pipeline_protocol(Path(args.ablation))
    comparability = comparability_report(baselines, pipeline)

    payload = {
        "_meta": {
            "script": "evaluation/run_baseline_audit.py",
            "commit": _git_commit(),
            "generated": datetime.now().isoformat(timespec="seconds"),
            "purpose": (
                "R1-8 fidelity validation and the post-E2 coverage extension. This is an audit "
                "because the neuro-symbolic baselines R1-8 refers to are not in this repository."
            ),
        },
        "neurosymbolic_baselines": SEARCH_EVIDENCE,
        "classical_baselines_found": baselines,
        "pipeline_protocol_for_comparison": pipeline,
        "comparability": comparability,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_scores.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 94)
    print("  E11 — BASELINE AUDIT")
    print("=" * 94)
    print("  Neuro-symbolic baselines (Table 12): NOT FOUND IN REPOSITORY")
    print(f"    {SEARCH_EVIDENCE['what_was_found_instead'][:150]}...")
    print()
    print("  Classical baselines found (Table 2):")
    for b in baselines:
        s = b["scores"]
        p = b["protocol"]
        print(f"    {b['run_id']} {b['dataset']:<7} {b['model']:<24} "
              f"F1-w={s['f1_weighted']:.4f}  n_test={p['n_test']:<4} "
              f"n_feat={p['n_features']:<5} classes={p['n_classes']}  gate=none coverage=100%")
        for flag in b["degeneracy_flags"]:
            print(f"        !! {flag}")
    print()
    if comparability.get("available"):
        print("  Comparability vs the pipeline results:")
        for m in comparability["mismatches"]:
            print(f"    - {m}")
        print(f"    coverage_matched: {comparability['coverage_matched']}")
    print("=" * 94)

    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
