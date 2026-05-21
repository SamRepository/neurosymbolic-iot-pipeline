"""Generate the feedback-cycle ablation figure (Figure 6) and matching
LaTeX table.

The figure characterises the feedback loop's correction policy under a
controlled false-positive injection regime: three panels show weighted
F1, false-positive rate, and prediction correctness per feedback cycle
for CASAS and SPHERE.

Reference lines on each panel come from the deterministic 5-fold CV
result on CASAS Aruba (Table~11 of the paper) so the synthetic
mechanism-level curve is anchored to the held-out empirical claim:

    AI-Only      F1-w 83.4% | Cor 82.6% | FP 17.4%
    NeSy-Full    F1-w 85.0% | Cor 84.6% | FP 15.4%
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Aruba 5-fold CV reference values (evaluation/results_aruba/ablation_cv.json)
ARUBA_AI_F1 = 0.834
ARUBA_NESY_F1 = 0.850
ARUBA_AI_COR = 0.826
ARUBA_NESY_COR = 0.846
ARUBA_AI_FP = 0.174
ARUBA_NESY_FP = 0.154


def _axhref(ax, y, color, label, side="right", pad=0.005):
    """Horizontal reference line with a compact label at the panel edge."""
    ax.axhline(y, color=color, linestyle=":", linewidth=0.9, alpha=0.85, zorder=1)
    xlim = ax.get_xlim()
    x_text = xlim[1] - 0.15 if side == "right" else xlim[0] + 0.15
    ha = "right" if side == "right" else "left"
    ax.text(x_text, y + pad, label, fontsize=7, color=color,
            ha=ha, va="bottom", style="italic")


def main() -> int:
    results_path = Path(
        "outputs/experiments/feedback_cycle_ablation/feedback_cycle_ablation_results.json"
    )
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run the experiment first.")
        return 1

    data = json.loads(results_path.read_text())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    casas = data["datasets"]["casas"]["aggregated"]
    sphere = data["datasets"]["sphere"]["aggregated"]

    cycles_c = np.array([a["cycle"] for a in casas])
    cycles_s = np.array([a["cycle"] for a in sphere])

    f1_c = np.array([a["f1_weighted_mean"] for a in casas])
    f1_c_std = np.array([a["f1_weighted_std"] for a in casas])
    fp_c = np.array([a["fp_rate_mean"] for a in casas])
    fp_c_std = np.array([a["fp_rate_std"] for a in casas])
    cor_c = np.array([a["correctness_mean"] for a in casas])
    cor_c_std = np.array([a["correctness_std"] for a in casas])

    f1_s = np.array([a["f1_weighted_mean"] for a in sphere])
    f1_s_std = np.array([a["f1_weighted_std"] for a in sphere])
    fp_s = np.array([a["fp_rate_mean"] for a in sphere])
    fp_s_std = np.array([a["fp_rate_std"] for a in sphere])
    cor_s = np.array([a["correctness_mean"] for a in sphere])
    cor_s_std = np.array([a["correctness_std"] for a in sphere])

    # Q1-journal-friendly figure: tight axes, no orphan reference lines,
    # publication-grade typography. 3 panels in a single row.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    c_casas = "#1F77B4"
    c_sphere = "#D9822B"
    c_ai = "#7F7F7F"
    c_full = "#2CA02C"

    def _series(ax, x, y, std, marker, color, label):
        ax.plot(x, y, marker + "-", color=color, label=label,
                linewidth=1.8, markersize=5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3)
        ax.fill_between(x, np.maximum(0.0, y - std), np.minimum(1.0, y + std),
                        color=color, alpha=0.18, linewidth=0, zorder=2)

    # ---- (a) Weighted F1 ----
    ax = axes[0]
    _series(ax, cycles_c, f1_c, f1_c_std, "o", c_casas, "CASAS")
    _series(ax, cycles_s, f1_s, f1_s_std, "s", c_sphere, "SPHERE")
    ax.set_xlim(-0.2, 8.2)
    lo = min(f1_c.min() - f1_c_std.max(), f1_s.min() - f1_s_std.max(),
             ARUBA_AI_F1) - 0.03
    hi = max(f1_c.max() + f1_c_std.max(), f1_s.max() + f1_s_std.max(),
             ARUBA_NESY_F1) + 0.03
    ax.set_ylim(round(lo, 2), round(hi, 2))
    _axhref(ax, ARUBA_AI_F1, c_ai,
            f"Aruba AI-Only (held-out): {ARUBA_AI_F1:.3f}", side="right", pad=0.003)
    _axhref(ax, ARUBA_NESY_F1, c_full,
            f"Aruba NeSy-Full (held-out): {ARUBA_NESY_F1:.3f}", side="right", pad=0.003)
    ax.set_xlabel("Feedback Cycle", fontsize=10, fontweight="bold")
    ax.set_ylabel("Weighted F1", fontsize=10, fontweight="bold")
    ax.set_title("(a) Weighted F1 vs. Feedback Cycle",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(range(9))
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- (b) False-positive rate ----
    ax = axes[1]
    _series(ax, cycles_c, fp_c, fp_c_std, "o", c_casas, "CASAS")
    _series(ax, cycles_s, fp_s, fp_s_std, "s", c_sphere, "SPHERE")
    ax.set_xlim(-0.2, 8.2)
    lo = max(0.0, min(fp_c.min() - fp_c_std.max(), fp_s.min() - fp_s_std.max(),
                       ARUBA_NESY_FP) - 0.03)
    hi = max(fp_c.max() + fp_c_std.max(), fp_s.max() + fp_s_std.max(),
             ARUBA_AI_FP) + 0.03
    ax.set_ylim(round(lo, 2), round(hi, 2))
    _axhref(ax, ARUBA_AI_FP, c_ai,
            f"Aruba AI-Only (held-out): {ARUBA_AI_FP:.3f}", side="right", pad=0.003)
    _axhref(ax, ARUBA_NESY_FP, c_full,
            f"Aruba NeSy-Full (held-out): {ARUBA_NESY_FP:.3f}", side="right", pad=0.003)
    ax.set_xlabel("Feedback Cycle", fontsize=10, fontweight="bold")
    ax.set_ylabel("False-Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title("(b) False-Positive Rate vs. Feedback Cycle",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(range(9))
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- (c) Prediction correctness ----
    ax = axes[2]
    _series(ax, cycles_c, cor_c, cor_c_std, "o", c_casas, "CASAS")
    _series(ax, cycles_s, cor_s, cor_s_std, "s", c_sphere, "SPHERE")
    ax.set_xlim(-0.2, 8.2)
    lo = min(cor_c.min() - cor_c_std.max(), cor_s.min() - cor_s_std.max(),
             ARUBA_AI_COR) - 0.03
    hi = max(cor_c.max() + cor_c_std.max(), cor_s.max() + cor_s_std.max(),
             ARUBA_NESY_COR) + 0.03
    ax.set_ylim(round(lo, 2), round(hi, 2))
    _axhref(ax, ARUBA_AI_COR, c_ai,
            f"Aruba AI-Only (held-out): {ARUBA_AI_COR:.3f}", side="right", pad=0.003)
    _axhref(ax, ARUBA_NESY_COR, c_full,
            f"Aruba NeSy-Full (held-out): {ARUBA_NESY_COR:.3f}", side="right", pad=0.003)
    ax.set_xlabel("Feedback Cycle", fontsize=10, fontweight="bold")
    ax.set_ylabel("Prediction Correctness", fontsize=10, fontweight="bold")
    ax.set_title("(c) Prediction Correctness vs. Feedback Cycle",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(range(9))
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.5)

    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "Figure_6_feedback_cycle_ablation.pdf",
                format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure_6_feedback_cycle_ablation.png",
                format="png", bbox_inches="tight", dpi=300)
    print("Saved:", out_dir / "Figure_6_feedback_cycle_ablation.pdf")
    print("Saved:", out_dir / "Figure_6_feedback_cycle_ablation.png")
    plt.close(fig)

    # ---- LaTeX table ----
    print()
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Synthetic feedback-cycle ablation: weighted F1, "
          r"false-positive rate, and correctness per cycle.}")
    print(r"    \label{tab:feedback-cycle-ablation}")
    print(r"    \begin{tabular}{crrrrrr}")
    print(r"        \toprule")
    print(r"        & \multicolumn{3}{c}{\textbf{CASAS}} "
          r"& \multicolumn{3}{c}{\textbf{SPHERE}} \\")
    print(r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"        \textbf{Cycle} & \textbf{F1} & \textbf{FP} & "
          r"\textbf{Corr.} & \textbf{F1} & \textbf{FP} & \textbf{Corr.} \\")
    print(r"        \midrule")
    for ca, sa in zip(casas, sphere):
        cyc = ca["cycle"]
        line = (f"        {cyc} & "
                f"{ca['f1_weighted_mean']:.3f} & "
                f"{ca['fp_rate_mean']:.3f} & "
                f"{ca['correctness_mean']:.3f} & "
                f"{sa['f1_weighted_mean']:.3f} & "
                f"{sa['fp_rate_mean']:.3f} & "
                f"{sa['correctness_mean']:.3f}")
        print(line + r" \\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
