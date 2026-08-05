"""
Figure 7 — cross-dataset federated reasoning (E8 regeneration).

Regenerates both panels from the E1 20-trial data
(``evaluation/results_federation/multi_trial.json`` and ``per_category_prf.json``),
replacing the legacy 3-trial figure produced by ``generate_cross_dataset_federation.py``.

Changes from the legacy figure, each tied to a reviewer item:

* **R2-5 (panel b unreadable).** The legacy panel (b) plotted KG triples and anomaly
  counts on two different y-axes via ``twinx()``. A dual-axis chart makes the
  relationship between the series a function of two arbitrary scalings, which is
  precisely the "relationship between the plotted series is unclear" the reviewer
  reported. Panel (b) is now a single-axis per-category precision/recall chart.
* **R1-4 / R2-9 (variance, and "more detections is not better detection").** Panel (a)
  now shows mean ± σ over 20 randomized trials instead of a single 3-trial mean, and
  panel (b) reports detection *quality* per category rather than volume.
* **Aggregate-lift annotation dropped.** The legacy "+88 %" arrow is removed: E1 showed
  the figure it annotated was inflated by a double count, and the manuscript has
  retired aggregate-lift figures as effect-size claims.

Colour: the legacy house palette (#309CF5/#F9A629/#4CAF50) fails colour-vision-deficiency
separation — green↔orange ΔE 5.5 under protanopia, below the usable floor — so this figure
uses darker Okabe-Ito-derived hues that pass CVD, chroma and print-contrast checks. Hatching
is applied as a secondary encoding so the bars remain distinguishable in greyscale print.

Usage:
  python fig/generate_federation_figure7.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# CVD-safe, print-contrast-safe. Semantic mapping preserved from the legacy figure
# (blue = CASAS, orange = SPHERE, green = federated) so the figure stays recognisable.
COLOR_CASAS = "#0067A3"
COLOR_SPHERE = "#C77700"
COLOR_FEDERATED = "#00785A"
COLOR_PRECISION = "#6A3D9A"
COLOR_RECALL = "#C77700"

INK_PRIMARY = "#1A1A1A"
INK_MUTED = "#5C5C5C"
GRID = "#CCCCCC"

HATCH_CASAS = ""
HATCH_SPHERE = "///"
HATCH_FEDERATED = "..."

SHARED_CATEGORIES: List[Tuple[str, str]] = [
    ("low_confidence", "Low\nconfidence"),
    ("night_time_events", "Night-time\nevents"),
    ("multi_room_motion", "Multi-room\nmotion"),
]
CROSS_CATEGORIES: List[Tuple[str, str]] = [
    ("cross_source_temporal_overlap", "Temporal\noverlap"),
    ("cross_source_activity_posture_conflict", "Activity–posture\nconflict"),
    ("cross_source_location_conflict", "Location\nconflict"),
]

# Cross-source PRF keys as emitted by run_federation_rigor.py. The three shared
# categories are rule-definitional (P = R = 1 by construction) and are reported in
# the caption rather than plotted — see _panel_b.
CROSS_PRF_KEYS: List[Tuple[str, str]] = [
    ("cross_source_temporal_overlap", "Temporal\noverlap"),
    ("cross_source_activity_posture_conflict", "Activity–posture\nconflict"),
    ("cross_source_location_conflict_event_scoped", "Location\nconflict"),
]


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_MUTED,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def _mean_sd(trials: List[Dict[str, Any]], path: List[str], key: str) -> Tuple[float, float]:
    """Mean and population-sample SD of one per-category count across trials."""
    values: List[float] = []
    for t in trials:
        node: Any = t
        for step in path:
            node = node[step]
        values.append(float(node.get(key, 0)))
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _panel_a(ax: plt.Axes, trials: List[Dict[str, Any]]) -> None:
    """Per-category detections, mean ± σ over the 20 randomized trials."""
    labels = [lab for _, lab in SHARED_CATEGORIES] + [lab for _, lab in CROSS_CATEGORIES]
    n_shared = len(SHARED_CATEGORIES)

    casas_m, casas_s = [], []
    sphere_m, sphere_s = [], []
    fed_m, fed_s = [], []

    for key, _ in SHARED_CATEGORIES:
        m, s = _mean_sd(trials, ["as_published", "casas"], key)
        casas_m.append(m); casas_s.append(s)
        m, s = _mean_sd(trials, ["as_published", "sphere"], key)
        sphere_m.append(m); sphere_s.append(s)
        m, s = _mean_sd(trials, ["as_published", "federated"], key)
        fed_m.append(m); fed_s.append(s)

    for key, _ in CROSS_CATEGORIES:
        # Cross-source categories are undetectable by either single source: a
        # single-source deployment never joins the two graphs. Plotted as absent,
        # not as zero-height bars, to avoid implying a measured zero.
        casas_m.append(np.nan); casas_s.append(0.0)
        sphere_m.append(np.nan); sphere_s.append(0.0)
        m, s = _mean_sd(trials, ["as_published", "federated"], key)
        fed_m.append(m); fed_s.append(s)

    x = np.arange(len(labels))
    width = 0.26
    err_kw = dict(ecolor=INK_MUTED, capsize=2.5, elinewidth=0.8, capthick=0.8)

    ax.bar(x - width, casas_m, width, yerr=casas_s, label="CASAS only",
           color=COLOR_CASAS, hatch=HATCH_CASAS, edgecolor="white", linewidth=0.7,
           error_kw=err_kw)
    ax.bar(x, sphere_m, width, yerr=sphere_s, label="SPHERE only",
           color=COLOR_SPHERE, hatch=HATCH_SPHERE, edgecolor="white", linewidth=0.7,
           error_kw=err_kw)
    ax.bar(x + width, fed_m, width, yerr=fed_s, label="Federated",
           color=COLOR_FEDERATED, hatch=HATCH_FEDERATED, edgecolor="white", linewidth=0.7,
           error_kw=err_kw)

    ax.set_ylabel("Detections per trial (mean ± σ)", fontsize=9.5)
    ax.set_title("(a) Detections by category", fontsize=10, pad=24)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.tick_params(axis="both", labelsize=8, colors=INK_PRIMARY, length=3)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    divider = n_shared - 0.5
    ax.axvline(divider, color=INK_MUTED, linestyle=":", linewidth=0.9, alpha=0.7)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.06)

    # Group annotations sit above the axes so they cannot collide with the legend
    # or with the tallest bar.
    ax.text((n_shared - 1) / 2, 1.012, "Shared categories",
            transform=ax.get_xaxis_transform(which="grid"), ha="center", va="bottom",
            fontsize=7.6, color=INK_MUTED, style="italic", clip_on=False)
    ax.text(n_shared + 1, 1.012, "Cross-source only (no single-source counterpart)",
            transform=ax.get_xaxis_transform(which="grid"), ha="center", va="bottom",
            fontsize=7.6, color=INK_MUTED, style="italic", clip_on=False)

    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.95, edgecolor=GRID,
              borderpad=0.5, handlelength=1.6)


def _panel_b(ax: plt.Axes, prf: Dict[str, Any]) -> None:
    """Cross-source detection quality against the injected ground truth, with the
    single-source recall the federated graph is being compared against.

    Only the three cross-source categories are plotted. The three shared categories
    are rule-definitional properties of the generated stream and score exactly 1.0
    by construction; plotting six near-identical pairs of full-height bars would
    imply a comparison where none exists, so they are stated in the caption instead.
    The informative contrast is federated recall against single-source recall, which
    is zero in principle: a single-source deployment never joins the two graphs.
    """
    x = np.arange(len(CROSS_PRF_KEYS))
    width = 0.27
    labels = [lab for _, lab in CROSS_PRF_KEYS]

    precision, recall, f1_vals, fp_counts = [], [], [], []
    for key, _ in CROSS_PRF_KEYS:
        pooled = prf["categories"][key]["pooled"]
        precision.append(float(pooled["precision"]))
        recall.append(float(pooled["recall"]))
        f1_vals.append(float(pooled["f1"]))
        fp_counts.append(int(pooled["fp"]))

    single_source_recall = float(prf.get("single_source_recall_on_cross_source_categories", 0.0))

    ax.bar(x - width, precision, width,
           color=COLOR_PRECISION, edgecolor="white", linewidth=0.7)
    ax.bar(x, recall, width, hatch="///",
           color=COLOR_RECALL, edgecolor="white", linewidth=0.7)

    # Single-source recall is exactly zero, so a bar would have no height and
    # render as nothing at all — a legend swatch for it would promise a mark the
    # reader cannot find. It is drawn instead as an explicit rule on the axis,
    # which is visible, and the legend carries that same rule as its handle.
    for xi in x:
        ax.plot([xi + width - width / 2, xi + width + width / 2],
                [single_source_recall, single_source_recall],
                color=INK_MUTED, linewidth=2.6, solid_capstyle="butt",
                zorder=6, clip_on=False)
        ax.text(xi + width, single_source_recall + 0.028, "0.00", ha="center", va="bottom",
                fontsize=6.8, color=INK_MUTED)

    for xi, (p_val, r_val, f1, fp) in enumerate(zip(precision, recall, f1_vals, fp_counts)):
        ax.text(xi - width / 2, max(p_val, r_val) + 0.035,
                f"F1 {f1:.2f}" + (f"  ({fp} FP)" if fp else "  (0 FP)"),
                ha="center", va="bottom", fontsize=6.8, color=INK_MUTED)

    ax.set_ylabel("Score (pooled over 20 trials)", fontsize=9.5)
    ax.set_title("(b) Cross-source detection quality vs. single-source reach",
                 fontsize=10, pad=24)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="both", labelsize=8, colors=INK_PRIMARY, length=3)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(0.5, 1.012, "Scored against the injected ground-truth registry",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.6, color=INK_MUTED, style="italic", clip_on=False)

    # Handles are built explicitly so every entry matches a mark that is actually
    # drawn: two bar patches and the zero rule. Kept inside the axes to avoid
    # spending vertical space on a legend strip; the near-opaque background keeps
    # it readable where it sits over the bars.
    handles = [
        Patch(facecolor=COLOR_PRECISION, edgecolor="white", label="Federated — precision"),
        Patch(facecolor=COLOR_RECALL, edgecolor="white", hatch="///", label="Federated — recall"),
        Line2D([], [], color=INK_MUTED, linewidth=2.6,
               label="Single-source — recall (0.00)"),
    ]
    ax.legend(handles=handles, fontsize=7.6, loc="lower center", framealpha=0.95,
              edgecolor=GRID, borderpad=0.5, handlelength=1.6, ncol=1)


CAPTION = """Figure 7: Cross-dataset federated reasoning, over 20 randomized injection trials
(100 CASAS + 100 SPHERE background predictions, 10 injected cross-source anomalies per trial;
seeds 42 + 1000k).

(a) Mean detections per trial by anomaly category, with error bars showing one standard
deviation across the 20 trials. Blue bars are a CASAS-only deployment, orange a SPHERE-only
deployment, green the federated knowledge graph. The three categories left of the dotted
divider are detectable from a single source; the three to its right are cross-source
conflicts, which have no single-source counterpart because a single-source deployment never
joins the two graphs (shown as absent rather than as zero-height bars).

(b) Detection quality for the three cross-source categories, scored against the injected
ground-truth registry and pooled over the 20 trials: federated precision and recall, with the
resulting F1 and false-positive count printed above each pair, beside the recall a single-source
deployment achieves on the same categories. That single-source recall is 0.00 in principle
rather than empirically: the conflicts are defined over events originating in two different
sources, which a single-source pipeline never joins, so it cannot express the query at all. The
three shared categories of panel (a) are omitted here because they are rule-definitional
properties of the generated stream and score precision = recall = 1.00 by construction; they
characterise the generator, not detector skill, and plotting them would imply a comparison
where none exists.

Note: aggregate-lift percentages are deliberately not annotated on this figure. The
as-published aggregate counted cross-source sensor pairs under two categories simultaneously;
the corrected accounting is reported in the text and in
evaluation/results_federation/multi_trial.json."""


def main() -> int:
    fed_dir = Path("evaluation/results_federation")
    multi_path = fed_dir / "multi_trial.json"
    prf_path = fed_dir / "per_category_prf.json"
    for path in (multi_path, prf_path):
        if not path.exists():
            print(f"ERROR: {path} not found. Run evaluation/run_federation_rigor.py first.")
            return 1

    multi = json.loads(multi_path.read_text(encoding="utf-8"))
    prf = json.loads(prf_path.read_text(encoding="utf-8"))
    trials = multi["raw_trials"]

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    _panel_a(axes[0], trials)
    _panel_b(axes[1], prf)
    fig.tight_layout(w_pad=2.6)

    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "Figure_7_cross_dataset_federation.pdf"
    png_path = out_dir / "Figure_7_cross_dataset_federation.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)

    caption_path = out_dir / "Figure_7_caption.txt"
    caption_path.write_text(
        CAPTION + f"\n\nSource data: {multi_path.as_posix()}, {prf_path.as_posix()}\n"
        f"Trials: {len(trials)}; generated by fig/generate_federation_figure7.py\n",
        encoding="utf-8",
    )

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {caption_path}")
    print(f"Panels regenerated from {len(trials)} trials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
