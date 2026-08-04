"""
Latency by deployment tier (E5 - FGCS revision)
===============================================
Addresses reviewer item R1-11: the latency framing must be calibrated to the
actual setup. The reviewer's three specific points are that the CPU-only
benchmark uses a desktop-class processor representative of a fog node, that
single-window CASAS latency is substantially higher than the headline sub-5 ms
figure, and that sub-5 ms appears only at batch size 50 - so fog-tier batched
feasibility must be distinguished from strict edge real-time operation.

This script reports the *published* measurements (from the stored edge-latency
benchmark) re-presented by tier. It does not re-run the benchmark: re-running
would produce different timings on different hardware and would silently replace
the numbers the manuscript quotes.

The central distinction it makes explicit:

  * **Time-to-decision** (total end-to-end latency for one inference call) is what
    strict edge real-time operation is bound by. A window cannot be acted on
    before its batch completes.
  * **Amortised per-window cost** (end-to-end / batch size) is a *throughput*
    measure. It is the figure the sub-5 ms headline refers to, and it does not
    describe how quickly any individual window is answered.

Batching improves the second while leaving the first unchanged or slightly worse
- for CASAS the batch-50 end-to-end latency (201.33 ms) is marginally *higher*
than the single-window latency (198.66 ms). Reporting only the amortised figure
is what makes the system look edge-real-time when it is not.

Usage:
  PYTHONPATH=. python evaluation/run_latency_tiers.py
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# A window is "edge real-time" only if a decision is available inside this budget.
# 100 ms is the conventional interactive-response bound used for ambient systems;
# the manuscript should state whichever budget it adopts rather than leaving the
# reader to infer one from the sub-5 ms figure.
EDGE_REALTIME_BUDGET_MS = 100.0
SUB_5MS_CLAIM_MS = 5.0


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


def _tier_for(batch_size: int) -> str:
    """Deployment tier a given batch size corresponds to."""
    return "edge (single-window, strict real-time)" if batch_size == 1 else "fog (batched)"


def build_tier_table(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Re-present the stored measurements with explicit tier and metric semantics."""
    out: Dict[str, Any] = {}
    for dataset, rows in raw["datasets"].items():
        entries: List[Dict[str, Any]] = []
        for row in rows:
            batch = int(row["batch_size"])
            e2e = float(row["total_e2e_ms_mean"])
            per_window = float(row["per_window_ms_mean"])
            entries.append({
                "batch_size": batch,
                "tier": _tier_for(batch),
                "time_to_decision_ms": round(e2e, 2),
                "time_to_decision_ms_std": round(float(row["total_e2e_ms_std"]), 2),
                "amortised_per_window_ms": round(per_window, 2),
                "throughput_windows_per_sec": round(float(row["throughput_windows_per_sec_mean"]), 2),
                "meets_sub_5ms_amortised": per_window < SUB_5MS_CLAIM_MS,
                "meets_edge_realtime_budget": e2e <= EDGE_REALTIME_BUDGET_MS,
                "stage_breakdown_ms": {
                    "neural_inference": round(float(row["neural_inference_ms_mean"]), 2),
                    "kg_population": round(float(row["kg_population_ms_mean"]), 2),
                    "symbolic_reasoning": round(float(row["reasoning_ms_mean"]), 2),
                    "feedback": round(float(row["feedback_ms_mean"]), 2),
                },
                "peak_memory_mb": round(float(row["peak_memory_mb_mean"]), 1),
            })
        out[dataset] = entries
    return out


def build_findings(tiers: Dict[str, Any]) -> Dict[str, Any]:
    casas = {e["batch_size"]: e for e in tiers["casas"]}
    sphere = {e["batch_size"]: e for e in tiers["sphere"]}

    casas_sub5 = [b for b, e in sorted(casas.items()) if e["meets_sub_5ms_amortised"]]
    sphere_sub5 = [b for b, e in sorted(sphere.items()) if e["meets_sub_5ms_amortised"]]
    casas_realtime = [b for b, e in sorted(casas.items()) if e["meets_edge_realtime_budget"]]
    sphere_realtime = [b for b, e in sorted(sphere.items()) if e["meets_edge_realtime_budget"]]

    single = casas[1]
    batch50 = casas[50]

    return {
        "where_sub_5ms_actually_holds": {
            "statement": (
                f"Sub-5 ms is an amortised per-window figure, and on CASAS it is reached at exactly "
                f"one operating point: batch {casas_sub5[0] if casas_sub5 else 'none'} "
                f"({casas[50]['amortised_per_window_ms']:.2f} ms at batch 50). On SPHERE it is reached at "
                f"batch {sphere_sub5[0] if sphere_sub5 else 'none'} and above. At batch 1 the "
                f"amortised and true latencies coincide and are "
                f"{single['amortised_per_window_ms']:.2f} ms (CASAS) and "
                f"{sphere[1]['amortised_per_window_ms']:.2f} ms (SPHERE)."
            ),
            "casas_batch_sizes_meeting_sub_5ms": casas_sub5,
            "sphere_batch_sizes_meeting_sub_5ms": sphere_sub5,
        },
        "batching_does_not_reduce_time_to_decision": {
            "statement": (
                f"Batching buys throughput, not responsiveness. For CASAS the end-to-end latency of a "
                f"batch-50 call is {batch50['time_to_decision_ms']:.2f} ms, marginally *higher* than the "
                f"{single['time_to_decision_ms']:.2f} ms single-window call, because the amortised "
                f"figure divides a larger total across more windows. No window in a batch can be acted "
                f"on before the whole batch completes, so the sub-5 ms figure never describes how "
                f"quickly an individual event is answered."
            ),
            "casas_e2e_batch_1_ms": single["time_to_decision_ms"],
            "casas_e2e_batch_50_ms": batch50["time_to_decision_ms"],
            "delta_ms": round(batch50["time_to_decision_ms"] - single["time_to_decision_ms"], 2),
        },
        "throughput_figure_provenance": {
            "statement": (
                f"The headline throughput figure of ~497 windows/s is SPHERE at batch 50 "
                f"({sphere[50]['throughput_windows_per_sec']:.2f}/s). The corresponding CASAS figure is "
                f"{casas[50]['throughput_windows_per_sec']:.2f}/s - roughly half. If the manuscript quotes "
                f"497 windows/s it must name SPHERE, or a reader will attribute it to the CASAS "
                f"pipeline the ablation is built on."
            ),
            "sphere_batch50_throughput": sphere[50]["throughput_windows_per_sec"],
            "casas_batch50_throughput": casas[50]["throughput_windows_per_sec"],
        },
        "edge_realtime_feasibility": {
            "statement": (
                f"Against a {EDGE_REALTIME_BUDGET_MS:.0f} ms interactive budget, CASAS meets the bound at "
                f"batch sizes {casas_realtime or 'none'} and SPHERE at {sphere_realtime or 'none'}. "
                f"Single-window CASAS ({single['time_to_decision_ms']:.2f} ms) does not, so strict edge "
                f"real-time operation is not demonstrated for the CASAS pipeline on this hardware at any "
                f"batch size; what is demonstrated is fog-tier batched feasibility."
            ),
            "budget_ms": EDGE_REALTIME_BUDGET_MS,
            "casas_batch_sizes_within_budget": casas_realtime,
            "sphere_batch_sizes_within_budget": sphere_realtime,
        },
        "dominant_stage": {
            "statement": (
                f"At batch 1 the cost is dominated by symbolic reasoning, not by the network: "
                f"{single['stage_breakdown_ms']['symbolic_reasoning']:.2f} ms of "
                f"{single['time_to_decision_ms']:.2f} ms "
                f"({single['stage_breakdown_ms']['symbolic_reasoning'] / single['time_to_decision_ms'] * 100:.0f} %), "
                f"against {single['stage_breakdown_ms']['neural_inference']:.2f} ms of neural inference. "
                f"Any latency optimisation should target the rdflib SPARQL path first."
            ),
        },
        "arm_class_hardware": (
            "No ARM-class board (Raspberry Pi 4 / Jetson) was available for this revision, so no "
            "ARM measurement is reported. The manuscript should keep this as an explicit limitation "
            "and future-work item rather than extrapolating desktop CPU timings to edge hardware: "
            "the pipeline is dominated by an in-memory SPARQL engine whose performance on ARM is "
            "not predictable from these numbers."
        ),
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    tiers = payload["tiers"]
    f = payload["findings"]
    lines: List[str] = []
    lines.append("# Latency by deployment tier (E5 — reviewer item R1-11)")
    lines.append("")
    lines.append("**Two different quantities are reported per row and must not be conflated.**")
    lines.append("*Time-to-decision* is the end-to-end latency of one inference call — what strict")
    lines.append("edge real-time operation is bound by, since no window can be acted on before its")
    lines.append("batch completes. *Amortised per-window* is time-to-decision divided by batch size:")
    lines.append("a throughput measure, and the one the sub-5 ms headline refers to.")
    lines.append("")

    for dataset in ("casas", "sphere"):
        lines.append(f"## {dataset.upper()}")
        lines.append("")
        lines.append("| Batch | Tier | Time-to-decision (ms) | Amortised /window (ms) | Throughput (win/s) | Sub-5 ms? | ≤100 ms? |")
        lines.append("|---|---|---|---|---|---|---|")
        for e in tiers[dataset]:
            lines.append(
                f"| {e['batch_size']} | {e['tier']} | {e['time_to_decision_ms']:.2f} "
                f"| {e['amortised_per_window_ms']:.2f} | {e['throughput_windows_per_sec']:.2f} "
                f"| {'yes' if e['meets_sub_5ms_amortised'] else 'no'} "
                f"| {'yes' if e['meets_edge_realtime_budget'] else 'no'} |"
            )
        lines.append("")

    lines.append("## What the numbers do and do not support")
    lines.append("")
    for key in ("where_sub_5ms_actually_holds", "batching_does_not_reduce_time_to_decision",
                "throughput_figure_provenance", "edge_realtime_feasibility", "dominant_stage"):
        lines.append(f"- {f[key]['statement']}")
        lines.append("")

    lines.append("## ARM-class hardware")
    lines.append("")
    lines.append(f[  "arm_class_hardware"])
    lines.append("")
    lines.append("## Hardware provenance")
    lines.append("")
    lines.append(payload["hardware_provenance"]["statement"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5: latency re-presented by deployment tier.")
    p.add_argument("--source", default="outputs/experiments/edge_latency/edge_latency_results.json",
                   help="Stored edge-latency benchmark results")
    p.add_argument("--outdir", default="evaluation/results_latency")
    return p.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neurosymbolic_iot.utils.logging import setup_logging

    args = parse_args()
    setup_logging("INFO")

    source = Path(args.source)
    if not source.exists():
        log.error("Source benchmark not found: %s", source)
        log.error("It lives under outputs/, which is gitignored — re-run "
                  "evaluation/run_edge_latency.py or restore the file.")
        return 1

    raw = json.loads(source.read_text(encoding="utf-8"))
    tiers = build_tier_table(raw)
    findings = build_findings(tiers)

    stored_meta = raw.get("metadata", {})
    hardware_provenance = {
        "statement": (
            "The stored benchmark records device='%s' and torch='%s' but **does not record the CPU "
            "model**. The manuscript attributes these timings to an i7-12700K; that attribution "
            "rests on the authors' record, not on the artifact, and cannot be verified from it. "
            "The machine running this analysis is a %s, i.e. not the machine that produced the "
            "measurements — so the numbers were deliberately not regenerated here. Recommend adding "
            "CPU model capture to the benchmark's metadata so future runs are self-describing."
            % (stored_meta.get("device", "?"), stored_meta.get("torch_version", "?"),
               platform.processor() or platform.machine())
        ),
        "stored_metadata": stored_meta,
        "analysis_machine": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "note": "analysis host only — did not produce the measurements",
        },
    }

    payload = {
        "_meta": {
            "script": "evaluation/run_latency_tiers.py",
            "commit": _git_commit(),
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source": source.as_posix(),
            "method": (
                "Re-presents the stored measurements by deployment tier. No benchmark was re-run; "
                "re-running on different hardware would silently replace the published numbers."
            ),
            "definitions": {
                "time_to_decision_ms": (
                    "total end-to-end latency of one inference call (all four pipeline stages). "
                    "Binds strict edge real-time operation."
                ),
                "amortised_per_window_ms": (
                    "time_to_decision divided by batch size. A throughput measure; does not "
                    "describe how quickly an individual window is answered."
                ),
                "edge_realtime_budget_ms": EDGE_REALTIME_BUDGET_MS,
            },
        },
        "tiers": tiers,
        "findings": findings,
        "hardware_provenance": hardware_provenance,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_tiers.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = out_dir / "latency_tiers.md"
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print()
    print("=" * 96)
    print("  E5 — LATENCY BY DEPLOYMENT TIER")
    print("=" * 96)
    for dataset in ("casas", "sphere"):
        print(f"  {dataset.upper()}")
        print(f"    {'batch':>5} {'tier':<38} {'decision(ms)':>13} {'/window(ms)':>12} {'win/s':>8} {'<5ms':>6} {'<=100ms':>8}")
        for e in tiers[dataset]:
            print(f"    {e['batch_size']:>5} {e['tier']:<38} {e['time_to_decision_ms']:>13.2f} "
                  f"{e['amortised_per_window_ms']:>12.2f} {e['throughput_windows_per_sec']:>8.1f} "
                  f"{('yes' if e['meets_sub_5ms_amortised'] else 'no'):>6} "
                  f"{('yes' if e['meets_edge_realtime_budget'] else 'no'):>8}")
    print()
    print("  " + findings["batching_does_not_reduce_time_to_decision"]["statement"][:300])
    print()
    print("  " + findings["throughput_figure_provenance"]["statement"][:300])
    print("=" * 96)

    log.info("Wrote %s and %s", out_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
