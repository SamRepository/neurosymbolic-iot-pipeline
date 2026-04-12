"""Generate cross-dataset federated reasoning figure."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> int:
    results_path = Path("outputs/experiments/cross_dataset_federation/cross_dataset_federation_results.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Anomaly detection comparison
    ax = axes[0]
    single_keys = ["low_confidence", "night_time_events", "multi_room_motion"]
    cross_keys = ["cross_source_temporal_overlap", "cross_source_activity_posture_conflict", "cross_source_location_conflict"]
    all_keys = single_keys + cross_keys
    labels = ["Low\nConf.", "Night\nEvents", "Multi-room\nMotion",
              "X-src\nTemporal", "X-src\nAct-Post.", "X-src\nLocation"]

    x = np.arange(len(all_keys))
    width = 0.25
    casas_vals = [ca.get(k, 0) for k in all_keys]
    sphere_vals = [sa.get(k, 0) for k in all_keys]
    fed_vals = [fa.get(k, 0) for k in all_keys]

    ax.bar(x - width, casas_vals, width, label="CASAS Only", color="#309CF5", alpha=0.85)
    ax.bar(x, sphere_vals, width, label="SPHERE Only", color="#F9A629", alpha=0.85)
    bars_f = ax.bar(x + width, fed_vals, width, label="Federated", color="#4CAF50", alpha=0.85)
    for bar, val in zip(bars_f, fed_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.axvline(2.5, color="#888", linestyle=":", linewidth=0.8, alpha=0.6)
    ylim_top = max(fed_vals) * 1.15
    ax.set_ylim(0, ylim_top)
    ax.text(1.0, ylim_top * 0.93, "Single-source", ha="center", fontsize=8, color="#666", style="italic")
    ax.text(4.0, ylim_top * 0.93, "Cross-source\n(fed. only)", ha="center", fontsize=8, color="#666", style="italic")
    ax.set_ylabel("Anomalies Detected", fontsize=10, fontweight="bold")
    ax.set_title("(a) Anomaly Detection: Single vs. Federated", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) KG size and anomaly totals
    ax = axes[1]
    categories = ["CASAS\nOnly", "SPHERE\nOnly", "Union\n(sum)", "Federated"]
    triples = [
        data["single_source"]["casas"]["triples_mean"],
        data["single_source"]["sphere"]["triples_mean"],
        data["single_source"]["casas"]["triples_mean"] + data["single_source"]["sphere"]["triples_mean"],
        data["federated"]["triples_mean"],
    ]
    anomaly_totals = [ca["total"], sa["total"], ca["total"] + sa["total"], fa["total"]]
    x2 = np.arange(len(categories))
    colors_bars = ["#309CF5", "#F9A629", "#9E9E9E", "#4CAF50"]
    ax.bar(x2 - 0.2, triples, 0.35, color=colors_bars, alpha=0.4)
    ax.set_ylabel("KG Size (triples)", fontsize=10, fontweight="bold")
    ax2 = ax.twinx()
    ax2.bar(x2 + 0.2, anomaly_totals, 0.35, color=colors_bars, alpha=0.85)
    ax2.set_ylabel("Total Anomalies Detected", fontsize=10, fontweight="bold")
    union_total = ca["total"] + sa["total"]
    fed_total = fa["total"]
    if union_total > 0:
        pct = (fed_total - union_total) / union_total * 100
        ax2.annotate(f"+{pct:.0f}%", xy=(3.2, fed_total), xytext=(3.4, fed_total * 0.7),
                     fontsize=11, fontweight="bold", color="#2E7D32",
                     arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5))
    ax.set_title("(b) KG Size and Detection Coverage", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="KG Triples"),
                       Patch(facecolor="gray", alpha=0.85, label="Anomalies")], fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=3.0)
    out_dir = Path("fig")
    fig.savefig(out_dir / "Figure_7_cross_dataset_federation.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure_7_cross_dataset_federation.png", format="png", bbox_inches="tight", dpi=300)
    print("Saved Figure 7")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
