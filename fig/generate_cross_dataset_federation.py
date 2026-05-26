"""Generate cross-dataset federated reasoning figure and LaTeX table."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> int:
    results_path = Path("outputs/experiments/cross_dataset_federation/cross_dataset_federation_results.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run the experiment first.")
        return 1

    data = json.loads(results_path.read_text())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
    })

    ca = data["single_source"]["casas"]["anomalies"]
    sa = data["single_source"]["sphere"]["anomalies"]
    fa = data["federated"]["anomalies"]

    # ---- Figure: 1 row x 2 subplots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Anomaly detection comparison - grouped bar chart
    ax = axes[0]
    single_keys = ["low_confidence", "night_time_events", "multi_room_motion"]
    cross_keys = ["cross_source_temporal_overlap", "cross_source_activity_posture_conflict", "cross_source_location_conflict"]
    all_keys = single_keys + cross_keys
    labels = [
        "Low\nConfidence",
        "Night-time\nEvents",
        "Multi-room\nMotion",
        "Cross-src\nTemporal",
        "Cross-src\nAct-Posture",
        "Cross-src\nLocation",
    ]

    x = np.arange(len(all_keys))
    width = 0.25

    casas_vals = [ca.get(k, 0) for k in all_keys]
    sphere_vals = [sa.get(k, 0) for k in all_keys]
    fed_vals = [fa.get(k, 0) for k in all_keys]

    bars_c = ax.bar(x - width, casas_vals, width, label="CASAS Only", color="#309CF5", alpha=0.85)
    bars_s = ax.bar(x, sphere_vals, width, label="SPHERE Only", color="#F9A629", alpha=0.85)
    bars_f = ax.bar(x + width, fed_vals, width, label="Federated", color="#4CAF50", alpha=0.85)

    # Value labels on federated bars
    for bar, val in zip(bars_f, fed_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Divider between single-source and cross-source anomalies
    ax.axvline(2.5, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)
    ylim = ax.get_ylim()
    ax.text(1.0, ylim[1] * 0.93, "Single-source", ha="center", fontsize=8, color="#666666", style="italic")
    ax.text(4.0, ylim[1] * 0.93, "Cross-source\n(federated only)", ha="center", fontsize=8, color="#666666", style="italic")

    ax.set_ylabel("Anomalies Detected", fontsize=10, fontweight="bold")
    ax.set_title("(a) Anomaly Detection: Single-Source vs. Federated", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88), framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) KG size and anomaly totals comparison
    ax = axes[1]
    categories = ["CASAS\nOnly", "SPHERE\nOnly", "Union\n(sum)", "Federated"]
    triples = [
        data["single_source"]["casas"]["triples_mean"],
        data["single_source"]["sphere"]["triples_mean"],
        data["single_source"]["casas"]["triples_mean"] + data["single_source"]["sphere"]["triples_mean"],
        data["federated"]["triples_mean"],
    ]
    anomaly_totals = [
        ca["total"],
        sa["total"],
        ca["total"] + sa["total"],
        fa["total"],
    ]

    x2 = np.arange(len(categories))
    colors_bars = ["#309CF5", "#F9A629", "#9E9E9E", "#4CAF50"]

    # Triples (left y-axis)
    bars_t = ax.bar(x2 - 0.2, triples, 0.35, color=colors_bars, alpha=0.4, label="KG Triples")
    ax.set_ylabel("KG Size (triples)", fontsize=10, fontweight="bold")

    # Anomalies (right y-axis)
    ax2 = ax.twinx()
    bars_a = ax2.bar(x2 + 0.2, anomaly_totals, 0.35, color=colors_bars, alpha=0.85, label="Anomalies")
    ax2.set_ylabel("Total Anomalies Detected", fontsize=10, fontweight="bold")

    # Improvement annotation
    union_total = ca["total"] + sa["total"]
    fed_total = fa["total"]
    if union_total > 0:
        pct = (fed_total - union_total) / union_total * 100
        ax2.annotate(
            f"+{pct:.0f}%",
            xy=(3.15, fed_total),
            xytext=(2.6, fed_total * 0.75),
            fontsize=11, fontweight="bold", color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5),
        )

    ax.set_title("(b) KG Size and Detection Coverage", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.4, label="KG Triples"),
        Patch(facecolor="gray", alpha=0.85, label="Anomalies"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    fig.tight_layout(w_pad=3.0)

    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "Figure_7_cross_dataset_federation.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure_7_cross_dataset_federation.png", format="png", bbox_inches="tight", dpi=300)
    print(f"Saved: {out_dir / 'Figure_7_cross_dataset_federation.pdf'}")
    print(f"Saved: {out_dir / 'Figure_7_cross_dataset_federation.png'}")
    plt.close(fig)

    # ---- LaTeX table ----
    print()
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Cross-dataset federated reasoning: anomaly detection improvement with unified KG.}")
    print(r"    \label{tab:cross-dataset-federation}")
    print(r"    \begin{tabular}{lrrr}")
    print(r"        \toprule")
    print(r"        \textbf{Anomaly Type} & \textbf{CASAS} & \textbf{SPHERE} & \textbf{Federated} \\")
    print(r"        \midrule")
    for key, label in [("low_confidence", "Low confidence"), ("night_time_events", "Night-time events"),
                       ("multi_room_motion", "Multi-room motion")]:
        print(f"        {label} & {ca[key]:.0f} & {sa[key]:.0f} & {fa[key]:.0f} \\\\")
    print(r"        \midrule")
    for key, label in [("cross_source_temporal_overlap", "Cross-src temporal overlap"),
                       ("cross_source_activity_posture_conflict", "Cross-src activity-posture"),
                       ("cross_source_location_conflict", "Cross-src location conflict")]:
        print(f"        {label} & --- & --- & {fa.get(key, 0):.0f} \\\\")
    print(r"        \midrule")
    print(f"        \\textbf{{Total}} & \\textbf{{{ca['total']:.0f}}} & \\textbf{{{sa['total']:.0f}}} & \\textbf{{{fa['total']:.0f}}} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
