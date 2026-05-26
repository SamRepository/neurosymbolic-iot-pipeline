"""Figure 6 — Ablation under 5-fold cross-validation on CASAS.

Reads ``evaluation/results/ablation_cv.json`` (output of
``evaluation/run_ablation_cv.py``) and renders a two-panel figure that
mirrors Table~11 in the manuscript:

* Panel (a): per-fold weighted F1 for the four ablation configurations,
  with the mean as a bar and individual fold values as overlaid points.
  The paired Wilcoxon $p$-value and Cohen's $d$ for the
  NeSy-Full vs AI-Only comparison are annotated above the NeSy-Full
  bar.
* Panel (b): correctness comparison. KG+Rules is measured on the
  rule-validated subset only (rules 3 and 4 fired), making it visually
  distinct (hatched) to signal selectivity rather than blanket coverage.

Output: ``fig/Figure_6_ablation_cv.pdf`` and ``.png``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


CONFIGS = ["AI-Only", "KG+Rules", "NeSy-NoFeedback", "NeSy-Full"]
BAR_COLORS = ["#9E9E9E", "#FFB300", "#42A5F5", "#1B5E20"]
LABEL_OVERRIDES = {
    "AI-Only": "AI-Only",
    "KG+Rules": "KG+Rules$^\\dagger$",
    "NeSy-NoFeedback": "NeSy-NoFB",
    "NeSy-Full": "NeSy-Full",
}


def _per_fold(per_fold_dict: Dict[str, Dict[str, List[Any]]],
              cfg: str, metric: str) -> List[float]:
    """Return the list of per-fold values for one (cfg, metric), with
    None entries (e.g. KG+Rules F1) coerced to NaN so matplotlib skips
    them cleanly."""
    vals = per_fold_dict.get(cfg, {}).get(metric, [])
    return [float(v) if v is not None else float("nan") for v in vals]


def _bar_with_points(ax, idx, vals, mean, sd, color, hatch=None, jitter=0.08):
    """Draw mean as a bar + individual fold points as dots."""
    bar = ax.bar(idx, mean, width=0.7, color=color, edgecolor="black",
                 linewidth=0.6, hatch=hatch, zorder=2)
    if not np.isnan(sd) and sd > 0:
        ax.errorbar(idx, mean, yerr=sd, fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=4, capthick=0.8, zorder=3)
    finite = [v for v in vals if not np.isnan(v)]
    if finite:
        rng = np.random.default_rng(idx * 17 + 3)
        xs = idx + (rng.random(len(finite)) - 0.5) * 2 * jitter
        ax.scatter(xs, finite, s=18, color="white", edgecolor="black",
                   linewidth=0.6, zorder=4)
    return bar


def main() -> int:
    results_path = Path("evaluation/results/ablation_cv.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. "
              f"Run evaluation/run_ablation_cv.py first.")
        return 1

    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    paired = data["paired_tests"]
    per_fold = data["per_fold"]

    # Pull stats for each configuration.
    means_f1: Dict[str, float] = {}
    sds_f1: Dict[str, float] = {}
    means_cor: Dict[str, float] = {}
    sds_cor: Dict[str, float] = {}
    for cfg in CONFIGS:
        s = summary.get(cfg, {})
        means_f1[cfg] = s.get("F1_weighted_mean") or float("nan")
        sds_f1[cfg] = s.get("F1_weighted_sd") or 0.0
        means_cor[cfg] = s.get("Correctness_mean") or float("nan")
        sds_cor[cfg] = s.get("Correctness_sd") or 0.0

    # Paired test for the headline annotation (NeSy-Full vs AI-Only on F1).
    p_full_vs_ai = paired.get("F1_weighted__AI-Only_vs_NeSy-Full", {})
    p_val = p_full_vs_ai.get("p_value")
    d_val = p_full_vs_ai.get("cohens_d")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
    })

    fig, (ax_f1, ax_cor) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                        gridspec_kw={"wspace": 0.30})

    # --- Panel (a): F1-weighted ---
    xs = np.arange(len(CONFIGS))
    for i, cfg in enumerate(CONFIGS):
        mean = means_f1[cfg]
        sd = sds_f1[cfg]
        if np.isnan(mean):
            # KG+Rules has no F1; draw a hatched placeholder bar.
            ax_f1.bar(i, 0.0, width=0.7, color="white",
                      edgecolor="0.5", hatch="//", linewidth=0.6, zorder=2)
            ax_f1.text(i, 2, "N/A", ha="center", va="bottom",
                       fontsize=9, color="0.4")
            continue
        vals = [v * 100 for v in _per_fold(per_fold, cfg, "F1_weighted")
                if not np.isnan(v)]
        _bar_with_points(ax_f1, i, vals, mean * 100, sd * 100,
                         color=BAR_COLORS[i])

    # NeSy-Full bar gets a paired-test annotation.
    if p_val is not None and not np.isnan(means_f1["NeSy-Full"]):
        sig = "n.s." if p_val >= 0.05 else "*"
        ann = (f"vs AI-Only\n$p$={p_val:.3f}, $d$={d_val:.2f}\n{sig}"
               if d_val is not None else f"$p$={p_val:.3f}")
        ax_f1.annotate(ann,
                       xy=(3, means_f1["NeSy-Full"] * 100 + sds_f1["NeSy-Full"] * 100),
                       xytext=(3, means_f1["NeSy-Full"] * 100 + sds_f1["NeSy-Full"] * 100 + 6),
                       ha="center", va="bottom", fontsize=7.5,
                       color="0.25",
                       arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))

    ax_f1.set_xticks(xs)
    ax_f1.set_xticklabels([LABEL_OVERRIDES[c] for c in CONFIGS], rotation=0,
                          fontsize=9)
    ax_f1.set_ylabel("F1-weighted (%)", fontsize=10, fontweight="bold")
    ax_f1.set_ylim(0, 105)
    ax_f1.set_title("(a) Held-out F1 across 5 stratified folds",
                    fontsize=10, fontweight="bold", pad=6)
    ax_f1.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax_f1.axhline(means_f1["AI-Only"] * 100, color="0.35", linestyle="--",
                  linewidth=0.7, alpha=0.7, zorder=1)
    ax_f1.text(0.02, means_f1["AI-Only"] * 100 + 1.5,
               "AI-Only baseline", transform=ax_f1.get_yaxis_transform(),
               fontsize=7.5, color="0.35")

    # --- Panel (b): Correctness ---
    for i, cfg in enumerate(CONFIGS):
        mean = means_cor[cfg]
        sd = sds_cor[cfg]
        if np.isnan(mean):
            ax_cor.bar(i, 0.0, width=0.7, color="white",
                       edgecolor="0.5", hatch="//", linewidth=0.6, zorder=2)
            ax_cor.text(i, 2, "N/A", ha="center", va="bottom",
                        fontsize=9, color="0.4")
            continue
        vals = [v * 100 for v in _per_fold(per_fold, cfg, "Correctness")
                if not np.isnan(v)]
        # KG+Rules is hatched to signal it's measured on a different
        # (rule-validated) subset, not the full held-out fold.
        hatch = "////" if cfg == "KG+Rules" else None
        _bar_with_points(ax_cor, i, vals, mean * 100, sd * 100,
                         color=BAR_COLORS[i], hatch=hatch)

    # Highlight the selectivity gain: KG+Rules vs AI-Only correctness.
    delta = (means_cor["KG+Rules"] - means_cor["AI-Only"]) * 100
    if not np.isnan(delta):
        kg_top = means_cor["KG+Rules"] * 100 + sds_cor["KG+Rules"] * 100
        ai_top = means_cor["AI-Only"] * 100 + sds_cor["AI-Only"] * 100
        bracket_top = max(kg_top, ai_top) + 4
        ax_cor.plot([0, 0, 1, 1],
                    [bracket_top - 1, bracket_top, bracket_top, bracket_top - 1],
                    color="black", linewidth=0.7)
        ax_cor.text(0.5, bracket_top + 0.5,
                    f"+{delta:.1f} pp (rule-validated subset)",
                    ha="center", va="bottom", fontsize=7.5, color="0.15")

    ax_cor.set_xticks(xs)
    ax_cor.set_xticklabels([LABEL_OVERRIDES[c] for c in CONFIGS], rotation=0,
                           fontsize=9)
    ax_cor.set_ylabel("Correctness (%)", fontsize=10, fontweight="bold")
    ax_cor.set_ylim(0, 105)
    ax_cor.set_title("(b) Correctness — full set vs rule-validated subset",
                     fontsize=10, fontweight="bold", pad=6)
    ax_cor.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

    # Shared footnote
    fig.text(0.5, 0.005,
             "$^\\dagger$ KG+Rules is evaluated on the rule-validated "
             "subset (rules 3--4 fired); F1 is undefined on this sparse subset.",
             ha="center", fontsize=7.5, color="0.30")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.90)

    out_dir = Path("fig")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "Figure_6_ablation_cv.pdf"
    png = out_dir / "Figure_6_ablation_cv.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"PDF: {pdf}")
    print(f"PNG: {png}")

    # Print a LaTeX caption block users can paste into the paper.
    print()
    print(r"% --- Suggested Figure 6 caption ---")
    print(r"\begin{figure*}[t]")
    print(r"\centering")
    print(r"\includegraphics[width=0.95\textwidth]{fig/Figure_6_ablation_cv.pdf}")
    print(rf"\caption{{Held-out 5-fold cross-validation ablation on CASAS "
          rf"(Seed = 42, $n_{{\text{{windows}}}}{{=}}170$). "
          rf"(a)~Weighted F1 per configuration with individual fold values "
          rf"overlaid; the paired Wilcoxon test for NeSy-Full vs AI-Only is "
          rf"not significant at $\alpha{{=}}0.05$ ($p{{=}}{p_val:.3f}$, "
          rf"Cohen's $d{{=}}{d_val:.2f}$). (b)~Correctness across "
          rf"configurations; the KG+Rules bar (hatched) is measured on the "
          rf"rule-validated subset only (25--50\% of windows per fold) and "
          rf"is +{delta:.1f}~pp above the AI-Only baseline, indicating "
          rf"strong selectivity rather than blanket gain.}}")
    print(r"\label{fig:feedback-cycle}")
    print(r"\end{figure*}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
