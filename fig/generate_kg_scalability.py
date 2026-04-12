"""Generate KG scalability figure (Figure 4) and LaTeX table."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> int:
    results_path = Path("outputs/experiments/kg_scalability/kg_scalability_results.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run evaluation/run_kg_scalability.py first.")
        return 1

    data = json.loads(results_path.read_text())
    datasets = data["datasets"]

    # ── Matplotlib style ────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
    })

    COLORS = {"casas": "#309CF5", "sphere": "#F9A629"}
    LABELS = {"casas": "CASAS", "sphere": "SPHERE"}

    # ── Extract arrays ──────────────────────────────────────────────
    def extract(dataset_name: str):
        entries = datasets[dataset_name]
        sizes = np.array([e["target_triples"] for e in entries])
        latency_mean = np.array([e["avg_query_latency_mean"] for e in entries])
        latency_std = np.array([e["avg_query_latency_std"] for e in entries])
        throughput_mean = np.array([e["throughput_mean"] for e in entries])
        throughput_std = np.array([e["throughput_std"] for e in entries])
        memory_mean = np.array([e["peak_memory_mean"] for e in entries])
        memory_std = np.array([e["peak_memory_std"] for e in entries])
        return sizes, latency_mean, latency_std, throughput_mean, throughput_std, memory_mean, memory_std

    casas = extract("casas")
    sphere = extract("sphere")
    sizes = casas[0]  # same for both

    # ── Figure: 1 row x 3 subplots ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) Query Latency vs KG Size
    ax = axes[0]
    for name, vals, color in [("casas", casas, COLORS["casas"]), ("sphere", sphere, COLORS["sphere"])]:
        ax.plot(vals[0], vals[1], "o-", color=color, label=LABELS[name],
                markersize=4, linewidth=1.5)
        ax.fill_between(vals[0], vals[1] - vals[2], vals[1] + vals[2],
                        color=color, alpha=0.15)
    ax.axhline(50, color="#D32F2F", linestyle="--", linewidth=1.0, alpha=0.7, label="50 ms threshold")
    ax.axvline(3000, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6, label="Paper range (3K)")
    ax.set_xlabel("KG Size (triples)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Query Latency (ms)", fontsize=10, fontweight="bold")
    ax.set_title("(a) Query Latency", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="both", alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    show = {500, 1000, 2000, 3000, 5000, 10000, 20000}
    ax.set_xticks(sizes)
    ax.set_xticklabels([(f"{s//1000}K" if s >= 1000 else str(s)) if s in show else "" for s in sizes],
                       fontsize=8, rotation=45, ha="right")

    # (b) Throughput vs KG Size
    ax = axes[1]
    for name, vals, color in [("casas", casas, COLORS["casas"]), ("sphere", sphere, COLORS["sphere"])]:
        ax.plot(vals[0], vals[3], "s-", color=color, label=LABELS[name],
                markersize=4, linewidth=1.5)
        ax.fill_between(vals[0], vals[3] - vals[4], vals[3] + vals[4],
                        color=color, alpha=0.15)
    ax.axvline(3000, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("KG Size (triples)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (triples/sec)", fontsize=10, fontweight="bold")
    ax.set_title("(b) Build Throughput", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="both", alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    show = {500, 1000, 2000, 3000, 5000, 10000, 20000}
    ax.set_xticks(sizes)
    ax.set_xticklabels([(f"{s//1000}K" if s >= 1000 else str(s)) if s in show else "" for s in sizes],
                       fontsize=8, rotation=45, ha="right")

    # (c) Memory vs KG Size
    ax = axes[2]
    for name, vals, color in [("casas", casas, COLORS["casas"]), ("sphere", sphere, COLORS["sphere"])]:
        ax.plot(vals[0], vals[5], "^-", color=color, label=LABELS[name],
                markersize=4, linewidth=1.5)
        ax.fill_between(vals[0], vals[5] - vals[6], vals[5] + vals[6],
                        color=color, alpha=0.15)
    ax.axvline(3000, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("KG Size (triples)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Peak Memory (MB)", fontsize=10, fontweight="bold")
    ax.set_title("(c) Memory Usage", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="both", alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    show = {500, 1000, 2000, 3000, 5000, 10000, 20000}
    ax.set_xticks(sizes)
    ax.set_xticklabels([(f"{s//1000}K" if s >= 1000 else str(s)) if s in show else "" for s in sizes],
                       fontsize=8, rotation=45, ha="right")

    fig.tight_layout(w_pad=3.0)

    # ── Save ────────────────────────────────────────────────────────
    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "Figure_5_query_latency_vs_triples.pdf",
                format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure_5_query_latency_vs_triples.png",
                format="png", bbox_inches="tight", dpi=300)
    print(f"Saved: {out_dir / 'Figure_5_query_latency_vs_triples.pdf'}")
    print(f"Saved: {out_dir / 'Figure_5_query_latency_vs_triples.png'}")
    plt.close(fig)

    # ── LaTeX table ─────────────────────────────────────────────────
    print()
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{KG scalability: SPARQL query latency, build throughput, and peak memory as KG size grows from 500 to 20{,}000 triples.}")
    print(r"    \label{tab:kg-scalability}")
    print(r"    \begin{tabular}{lrrrrrr}")
    print(r"        \toprule")
    print(r"        & \multicolumn{3}{c}{\textbf{CASAS}} & \multicolumn{3}{c}{\textbf{SPHERE}} \\")
    print(r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"        \textbf{Triples} & \textbf{Latency} & \textbf{Throughput} & \textbf{Memory} & \textbf{Latency} & \textbf{Throughput} & \textbf{Memory} \\")
    print(r"        & (ms) & (triples/s) & (MB) & (ms) & (triples/s) & (MB) \\")
    print(r"        \midrule")
    for i, size in enumerate(sizes):
        c = datasets["casas"][i]
        s = datasets["sphere"][i]
        size_str = f"{size:,}"
        print(f"        {size_str} & {c['avg_query_latency_mean']:.1f} & {c['throughput_mean']:,.0f} & {c['peak_memory_mean']:.1f}"
              f" & {s['avg_query_latency_mean']:.1f} & {s['throughput_mean']:,.0f} & {s['peak_memory_mean']:.1f} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
