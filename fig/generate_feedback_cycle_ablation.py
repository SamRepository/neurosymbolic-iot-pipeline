"""Generate feedback cycle ablation figure and LaTeX table."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> int:
    results_path = Path("outputs/experiments/feedback_cycle_ablation/feedback_cycle_ablation_results.json")
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

    casas = data["datasets"]["casas"]["aggregated"]
    sphere = data["datasets"]["sphere"]["aggregated"]

    cycles_c = [a["cycle"] for a in casas]
    cycles_s = [a["cycle"] for a in sphere]

    # Extract metrics
    f1_c = [a["f1_weighted_mean"] for a in casas]
    f1_c_std = [a["f1_weighted_std"] for a in casas]
    fp_c = [a["fp_rate_mean"] for a in casas]
    fp_c_std = [a["fp_rate_std"] for a in casas]
    cor_c = [a["correctness_mean"] for a in casas]
    cor_c_std = [a["correctness_std"] for a in casas]

    f1_s = [a["f1_weighted_mean"] for a in sphere]
    f1_s_std = [a["f1_weighted_std"] for a in sphere]
    fp_s = [a["fp_rate_mean"] for a in sphere]
    fp_s_std = [a["fp_rate_std"] for a in sphere]
    cor_s = [a["correctness_mean"] for a in sphere]
    cor_s_std = [a["correctness_std"] for a in sphere]

    # ---- Figure: 1 row x 3 subplots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors
    c_casas = "#309CF5"
    c_sphere = "#F9A629"

    # (a) F1-Weighted Score
    ax = axes[0]
    ax.plot(cycles_c, f1_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(cycles_c, np.array(f1_c) - np.array(f1_c_std), np.array(f1_c) + np.array(f1_c_std), color=c_casas, alpha=0.15)
    ax.plot(cycles_s, f1_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(cycles_s, np.array(f1_s) - np.array(f1_s_std), np.array(f1_s) + np.array(f1_s_std), color=c_sphere, alpha=0.15)
    ax.axhline(0.894, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(6, 0.884, "Paper NS-Full F1=0.894", fontsize=7, color="#666666", style="italic")
    ax.set_xlabel("Feedback Cycle", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1-Weighted Score", fontsize=10, fontweight="bold")
    ax.set_title("(a) F1-Weighted Score vs. Feedback Cycle", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(range(9))
    ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) False Positive Rate
    ax = axes[1]
    ax.plot(cycles_c, fp_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(cycles_c, np.maximum(0, np.array(fp_c) - np.array(fp_c_std)), np.array(fp_c) + np.array(fp_c_std), color=c_casas, alpha=0.15)
    ax.plot(cycles_s, fp_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(cycles_s, np.maximum(0, np.array(fp_s) - np.array(fp_s_std)), np.array(fp_s) + np.array(fp_s_std), color=c_sphere, alpha=0.15)
    ax.axhline(0.05, color="#E53935", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(5.5, 0.058, "5% target", fontsize=7, color="#E53935", style="italic")
    # Convergence annotations
    ax.annotate("CASAS converges", xy=(5, fp_c[5]), xytext=(6.5, 0.08),
                fontsize=7, color=c_casas, arrowprops=dict(arrowstyle="->", color=c_casas, lw=1.0))
    ax.annotate("SPHERE converges", xy=(4, fp_s[4]), xytext=(5.5, 0.12),
                fontsize=7, color=c_sphere, arrowprops=dict(arrowstyle="->", color=c_sphere, lw=1.0))
    ax.set_xlabel("Feedback Cycle", fontsize=10, fontweight="bold")
    ax.set_ylabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title("(b) False Positive Rate vs. Feedback Cycle", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(range(9))
    ax.set_ylim(-0.01, 0.28)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) Correctness
    ax = axes[2]
    ax.plot(cycles_c, cor_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(cycles_c, np.array(cor_c) - np.array(cor_c_std), np.minimum(1.0, np.array(cor_c) + np.array(cor_c_std)), color=c_casas, alpha=0.15)
    ax.plot(cycles_s, cor_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(cycles_s, np.array(cor_s) - np.array(cor_s_std), np.minimum(1.0, np.array(cor_s) + np.array(cor_s_std)), color=c_sphere, alpha=0.15)
    ax.set_xlabel("Feedback Cycle", fontsize=10, fontweight="bold")
    ax.set_ylabel("Correctness", fontsize=10, fontweight="bold")
    ax.set_title("(c) Prediction Correctness vs. Feedback Cycle", fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(range(9))
    ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=3.0)

    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "Figure_6_feedback_cycle_ablation.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure_6_feedback_cycle_ablation.png", format="png", bbox_inches="tight", dpi=300)
    print("Saved:", out_dir / "Figure_6_feedback_cycle_ablation.pdf")
    print("Saved:", out_dir / "Figure_6_feedback_cycle_ablation.png")
    plt.close(fig)

    # ---- LaTeX table ----
    print()
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Feedback cycle ablation: F1-weighted, FP rate, and correctness per cycle.}")
    print(r"    \label{tab:feedback-cycle-ablation}")
    print(r"    \begin{tabular}{crrrrrr}")
    print(r"        \toprule")
    print(r"        & \multicolumn{3}{c}{\textbf{CASAS}} & \multicolumn{3}{c}{\textbf{SPHERE}} \\")
    print(r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"        \textbf{Cycle} & \textbf{F1} & \textbf{FP Rate} & \textbf{Correct.} & \textbf{F1} & \textbf{FP Rate} & \textbf{Correct.} \\")
    print(r"        \midrule")
    for ca, sa in zip(casas, sphere):
        cyc = ca["cycle"]
        line = "        %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f" % (cyc, ca["f1_weighted_mean"], ca["fp_rate_mean"], ca["correctness_mean"], sa["f1_weighted_mean"], sa["fp_rate_mean"], sa["correctness_mean"])
        print(line + r" \\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

