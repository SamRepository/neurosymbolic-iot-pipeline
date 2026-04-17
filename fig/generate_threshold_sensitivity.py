"""Generate confidence threshold sensitivity figure and LaTeX table."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> int:
    results_path = Path("outputs/experiments/threshold_sensitivity/threshold_sensitivity_results.json")
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

    thresh_c = [a["threshold"] for a in casas]
    thresh_s = [a["threshold"] for a in sphere]

    # Extract metrics
    fp_c = [a["fp_rate_mean"] for a in casas]
    fp_c_std = [a["fp_rate_std"] for a in casas]
    cor_c = [a["correctness_mean"] for a in casas]
    cor_c_std = [a["correctness_std"] for a in casas]
    fb_c = [a["feedback_triggers_mean"] for a in casas]
    fb_c_std = [a["feedback_triggers_std"] for a in casas]
    cov_c = [a["coverage_mean"] for a in casas]
    cov_c_std = [a["coverage_std"] for a in casas]

    fp_s = [a["fp_rate_mean"] for a in sphere]
    fp_s_std = [a["fp_rate_std"] for a in sphere]
    cor_s = [a["correctness_mean"] for a in sphere]
    cor_s_std = [a["correctness_std"] for a in sphere]
    fb_s = [a["feedback_triggers_mean"] for a in sphere]
    fb_s_std = [a["feedback_triggers_std"] for a in sphere]
    cov_s = [a["coverage_mean"] for a in sphere]
    cov_s_std = [a["coverage_std"] for a in sphere]

    c_casas = "#309CF5"
    c_sphere = "#F9A629"

    # ---- Figure: 2x2 subplots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) FP Rate vs Threshold
    ax = axes[0, 0]
    ax.plot(thresh_c, fp_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_c, np.maximum(0, np.array(fp_c) - np.array(fp_c_std)), np.array(fp_c) + np.array(fp_c_std), color=c_casas, alpha=0.15)
    ax.plot(thresh_s, fp_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_s, np.maximum(0, np.array(fp_s) - np.array(fp_s_std)), np.array(fp_s) + np.array(fp_s_std), color=c_sphere, alpha=0.15)
    ax.axvline(0.85, color="#E53935", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.855, max(fp_c) * 0.7, "Paper default", fontsize=7, color="#E53935", rotation=90, va="center")
    ax.set_xlabel("Confidence Threshold", fontsize=10, fontweight="bold")
    ax.set_ylabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title("(a) FP Rate vs. Confidence Threshold", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Correctness vs Threshold
    ax = axes[0, 1]
    ax.plot(thresh_c, cor_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_c, np.array(cor_c) - np.array(cor_c_std), np.minimum(1.0, np.array(cor_c) + np.array(cor_c_std)), color=c_casas, alpha=0.15)
    ax.plot(thresh_s, cor_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_s, np.array(cor_s) - np.array(cor_s_std), np.minimum(1.0, np.array(cor_s) + np.array(cor_s_std)), color=c_sphere, alpha=0.15)
    ax.axvline(0.85, color="#E53935", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.855, 0.88, "Paper default", fontsize=7, color="#E53935", rotation=90, va="center")
    ax.set_xlabel("Confidence Threshold", fontsize=10, fontweight="bold")
    ax.set_ylabel("Correctness", fontsize=10, fontweight="bold")
    ax.set_title("(b) Correctness vs. Confidence Threshold", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0.80, 1.02)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) FeedbackRequired Triggers vs Threshold
    ax = axes[1, 0]
    ax.plot(thresh_c, fb_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_c, np.maximum(0, np.array(fb_c) - np.array(fb_c_std)), np.array(fb_c) + np.array(fb_c_std), color=c_casas, alpha=0.15)
    ax.plot(thresh_s, fb_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_s, np.maximum(0, np.array(fb_s) - np.array(fb_s_std)), np.array(fb_s) + np.array(fb_s_std), color=c_sphere, alpha=0.15)
    ax.axvline(0.85, color="#E53935", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.855, max(fb_c) * 0.5, "Paper default", fontsize=7, color="#E53935", rotation=90, va="center")
    ax.set_xlabel("Confidence Threshold", fontsize=10, fontweight="bold")
    ax.set_ylabel("FeedbackRequired Triggers", fontsize=10, fontweight="bold")
    ax.set_title("(c) Feedback Triggers vs. Confidence Threshold", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (d) Coverage vs Threshold (tradeoff panel)
    ax = axes[1, 1]
    ax.plot(thresh_c, cov_c, "o-", color=c_casas, label="CASAS", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_c, np.maximum(0, np.array(cov_c) - np.array(cov_c_std)), np.minimum(1.0, np.array(cov_c) + np.array(cov_c_std)), color=c_casas, alpha=0.15)
    ax.plot(thresh_s, cov_s, "s-", color=c_sphere, label="SPHERE", linewidth=1.8, markersize=5)
    ax.fill_between(thresh_s, np.maximum(0, np.array(cov_s) - np.array(cov_s_std)), np.minimum(1.0, np.array(cov_s) + np.array(cov_s_std)), color=c_sphere, alpha=0.15)
    ax.axvline(0.85, color="#E53935", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.855, 0.6, "Paper default", fontsize=7, color="#E53935", rotation=90, va="center")
    # Shade the optimal zone
    ax.axvspan(0.80, 0.90, alpha=0.08, color="#4CAF50", label="Optimal zone")
    ax.set_xlabel("Confidence Threshold", fontsize=10, fontweight="bold")
    ax.set_ylabel("Validation Coverage", fontsize=10, fontweight="bold")
    ax.set_title("(d) Coverage vs. Confidence Threshold", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(h_pad=3.0, w_pad=3.0)

    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "Figure_9_threshold_sensitivity.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure_9_threshold_sensitivity.png", format="png", bbox_inches="tight", dpi=300)
    print("Saved:", out_dir / "Figure_9_threshold_sensitivity.pdf")
    print("Saved:", out_dir / "Figure_9_threshold_sensitivity.png")
    plt.close(fig)

    # ---- LaTeX table ----
    print()
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Confidence threshold sensitivity: impact on FP rate, correctness, feedback triggers, and coverage.}")
    print(r"    \label{tab:threshold-sensitivity}")
    print(r"    \begin{tabular}{crrrrrrrr}")
    print(r"        \toprule")
    print(r"        & \multicolumn{4}{c}{\textbf{CASAS}} & \multicolumn{4}{c}{\textbf{SPHERE}} \\")
    print(r"        \cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    print(r"        \textbf{Thresh.} & \textbf{FP} & \textbf{Corr.} & \textbf{Feedb.} & \textbf{Cov.} & \textbf{FP} & \textbf{Corr.} & \textbf{Feedb.} & \textbf{Cov.} \\")
    print(r"        \midrule")
    for ca, sa in zip(casas, sphere):
        t = ca["threshold"]
        line = "        %.2f & %.3f & %.3f & %.0f & %.3f & %.3f & %.3f & %.0f & %.3f" % (
            t, ca["fp_rate_mean"], ca["correctness_mean"], ca["feedback_triggers_mean"], ca["coverage_mean"],
            sa["fp_rate_mean"], sa["correctness_mean"], sa["feedback_triggers_mean"], sa["coverage_mean"])
        bold = ""
        if t == 0.85:
            bold = " *"
        print(line + r" \\" + bold)
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \vspace{1mm}")
    print(r"    \footnotesize{* Paper default threshold (0.85).}")
    print(r"\end{table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

