"""
Generate Figure 10: Edge Deployment Latency Benchmark
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path("outputs/experiments/edge_latency/edge_latency_results.json")
OUT_DIR = Path("fig")

STAGE_KEYS = ["neural_inference_ms", "kg_population_ms", "reasoning_ms", "feedback_ms"]
STAGE_LABELS = ["Neural Inference", "KG Population", "SPARQL Reasoning", "Feedback Detection"]
STAGE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

DS_LABELS = {"casas": "CASAS (GRU)", "sphere": "SPHERE (LSTM)"}


def main() -> int:
    if not RESULTS_PATH.exists():
        print(f"Results not found: {RESULTS_PATH}")
        return 1

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- (a) Stacked bar: per-stage latency breakdown ---
    ax = axes[0]
    datasets = ["casas", "sphere"]
    batch_sizes = [r["batch_size"] for r in results["datasets"]["casas"]]
    n_bars = len(batch_sizes)
    width = 0.35
    x = np.arange(n_bars)

    for di, ds in enumerate(datasets):
        ds_data = results["datasets"][ds]
        offset = (di - 0.5) * width
        bottom = np.zeros(n_bars)
        for si, skey in enumerate(STAGE_KEYS):
            vals = np.array([r[skey + "_mean"] for r in ds_data])
            label = STAGE_LABELS[si] if di == 0 else None
            ax.bar(x + offset, vals, width, bottom=bottom,
                   color=STAGE_COLORS[si], label=label, alpha=0.85)
            bottom += vals

    ax.set_xlabel("Batch Size (windows)")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("(a) Per-Stage Latency Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend(fontsize=8, loc="upper left")

    # Dataset labels on top bars
    for di, ds in enumerate(datasets):
        offset = (di - 0.5) * width
        ds_data = results["datasets"][ds]
        totals = [r["total_e2e_ms_mean"] for r in ds_data]
        for xi, total in enumerate(totals):
            if xi == n_bars - 1:
                lbl = DS_LABELS[ds].split(" ")[0]
                ax.text(xi + offset, total + 3, lbl, ha="center", fontsize=6)

    # --- (b) Per-window latency ---
    ax = axes[1]
    for ds in datasets:
        ds_data = results["datasets"][ds]
        bs_arr = [r["batch_size"] for r in ds_data]
        pw_mean = [r["per_window_ms_mean"] for r in ds_data]
        pw_std = [r["per_window_ms_std"] for r in ds_data]
        ax.errorbar(bs_arr, pw_mean, yerr=pw_std, marker="o", capsize=3,
                    label=DS_LABELS[ds], linewidth=2)

    ax.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="50ms real-time bound")
    ax.set_xlabel("Batch Size (windows)")
    ax.set_ylabel("Per-Window Latency (ms)")
    ax.set_title("(b) Per-Window Latency vs Batch Size")
    ax.legend(fontsize=8)
    ax.set_xticks(batch_sizes)

    # --- (c) Throughput ---
    ax = axes[2]
    for ds in datasets:
        ds_data = results["datasets"][ds]
        bs_arr = [r["batch_size"] for r in ds_data]
        tp_mean = [r["throughput_windows_per_sec_mean"] for r in ds_data]
        tp_std = [r["throughput_windows_per_sec_std"] for r in ds_data]
        ax.errorbar(bs_arr, tp_mean, yerr=tp_std, marker="s", capsize=3,
                    label=DS_LABELS[ds], linewidth=2)

    ax.set_xlabel("Batch Size (windows)")
    ax.set_ylabel("Throughput (windows/sec)")
    ax.set_title("(c) Edge Throughput vs Batch Size")
    ax.legend(fontsize=8)
    ax.set_xticks(batch_sizes)

    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out = OUT_DIR / f"Figure_10_edge_latency{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)

    plt.close(fig)

    _print_latex_table(results)

    return 0


def _print_latex_table(results):
    print()
    print("%% LaTeX table for edge deployment latency")
    print("{bs}begin{{table}}[ht]".format(bs="\\"))
    print("{bs}centering".format(bs="\\"))
    print("{bs}caption{{Edge Deployment Latency Benchmark (CPU-only)}}".format(bs="\\"))
    print("{bs}label{{tab:edge_latency}}".format(bs="\\"))
    print("{bs}begin{{tabular}}{{l r r r r r r}}".format(bs="\\"))
    print("{bs}hline".format(bs="\\"))
    print("Dataset & Batch & Neural & KG Pop. & Reasoning & Total E2E & Per-Win {bs}{bs}".format(bs="\\"))
    print("{bs}hline".format(bs="\\"))

    for ds in ["casas", "sphere"]:
        ds_data = results["datasets"][ds]
        ds_label = DS_LABELS[ds]
        for i, row in enumerate(ds_data):
            bs = row["batch_size"]
            n = row["neural_inference_ms_mean"]
            k = row["kg_population_ms_mean"]
            r = row["reasoning_ms_mean"]
            t = row["total_e2e_ms_mean"]
            pw = row["per_window_ms_mean"]
            prefix = ds_label if i == 0 else ""
            line = f"{prefix} & {bs} & {n:.1f} & {k:.1f} & {r:.1f} & {t:.1f} & {pw:.1f}"
            print(line + " {bs}{bs}".format(bs="\\"))
        print("{bs}hline".format(bs="\\"))

    print("{bs}end{{tabular}}".format(bs="\\"))
    print("{bs}end{{table}}".format(bs="\\"))


if __name__ == "__main__":
    raise SystemExit(main())
