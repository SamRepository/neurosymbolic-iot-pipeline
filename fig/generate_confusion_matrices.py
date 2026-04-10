"""Generate Figure 3: Confusion matrices for CASAS and SPHERE datasets.

Matrices are derived from aggregated 5-fold cross-validation predictions.
CASAS: GRU classifier, 4 activity classes, 170 windows.
SPHERE: LSTM classifier, 7 activity/posture classes, 139 windows.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── CASAS confusion matrix (4 classes, 170 windows, aggregated 5-fold CV) ──
casas_labels = ["Clean", "Cook", "Eat", "Phone\nCall"]
casas_cm = np.array([
    [83,  5,  4,  3],
    [ 9, 25,  4,  2],
    [ 5,  3, 12,  5],
    [ 5,  3,  1,  1],
])

# ── SPHERE confusion matrix (7 classes, 139 windows, aggregated 5-fold CV) ──
sphere_labels = [
    "Lying", "Sitting", "Standing", "Walking",
    "Transition", "Bending", "Kneeling",
]
sphere_cm = np.array([
    [ 5,  3,  1,  0,  1,  0,  0],
    [ 2, 14,  4,  1,  2,  1,  1],
    [ 1,  8, 11,  3,  4,  2,  1],
    [ 0,  3,  5,  6,  3,  2,  1],
    [ 1,  5,  6,  3,  4,  2,  1],
    [ 0,  3,  4,  2,  3,  3,  1],
    [ 0,  2,  4,  2,  3,  2,  3],
])


def plot_cm(ax, cm, labels, title, subtitle, cmap):
    """Plot a single normalized confusion matrix."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0,
                        out=np.zeros_like(cm, dtype=float))
    n = len(labels)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=1.0, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            fs = 6.5 if n > 5 else 8
            ax.text(j, i, f"{val:.0%}\n({count})", ha="center", va="center",
                    color=color, fontsize=fs, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick_fs = 7 if n > 5 else 8
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.set_xlabel("Predicted Label", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=8.5, fontweight="bold")
    # Title + subtitle with line break
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, fontweight="normal",
                 pad=6, linespacing=1.6)
    # Make the first line bold via a manual override — draw title parts separately
    # Actually use the built-in: first line bold, second italic
    ax.title.set_fontsize(9)
    return im


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
    })

    # Narrower gap between panels
    fig = plt.figure(figsize=(8.2, 4.4))
    gs = GridSpec(1, 5, figure=fig,
                  width_ratios=[1, 0.04, 0.18, 1, 0.04],
                  wspace=0.08)

    ax_casas   = fig.add_subplot(gs[0, 0])
    cax_casas  = fig.add_subplot(gs[0, 1])
    ax_sphere  = fig.add_subplot(gs[0, 3])
    cax_sphere = fig.add_subplot(gs[0, 4])

    casas_acc = np.trace(casas_cm) / casas_cm.sum()
    sphere_acc = np.trace(sphere_cm) / sphere_cm.sum()

    # (a) CASAS on the LEFT
    im1 = plot_cm(ax_casas, casas_cm, casas_labels,
                  "(a) CASAS \u2014 GRU",
                  f"Acc: {casas_acc:.1%} | F1-macro: 54.20%",
                  "Blues")
    # (b) SPHERE on the RIGHT
    im2 = plot_cm(ax_sphere, sphere_cm, sphere_labels,
                  "(b) SPHERE \u2014 LSTM",
                  f"Acc: {sphere_acc:.1%} | F1-macro: 22.48%",
                  "YlOrRd")

    # Make title line 1 bold, line 2 italic gray
    for ax in (ax_casas, ax_sphere):
        txt = ax.title
        txt.set_fontweight("normal")

    # Manually draw two-line titles for proper styling
    for ax, line1, line2 in [
        (ax_casas, "(a) CASAS \u2014 GRU",
         f"Acc: {casas_acc:.1%} | F1-macro: 54.20%"),
        (ax_sphere, "(b) SPHERE \u2014 LSTM",
         f"Acc: {sphere_acc:.1%} | F1-macro: 22.48%"),
    ]:
        ax.set_title("")  # clear default
        ax.text(0.5, 1.10, line1, transform=ax.transAxes, ha="center",
                fontsize=9.5, fontweight="bold")
        ax.text(0.5, 1.04, line2, transform=ax.transAxes, ha="center",
                fontsize=7.5, style="italic", color="0.35")

    # Individual colorbars
    cb1 = fig.colorbar(im1, cax=cax_casas)
    cb1.ax.tick_params(labelsize=6.5)

    cb2 = fig.colorbar(im2, cax=cax_sphere)
    cb2.set_label("Recall", fontsize=8, labelpad=8)
    cb2.ax.tick_params(labelsize=6.5)

    # Drop duplicate y-label on SPHERE
    ax_sphere.set_ylabel("")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.88)

    out_dir = Path(__file__).parent
    out_pdf = out_dir / "Figure_3_confusion_matrices_CASAS_SPHERE_upd.pdf"
    out_png = out_dir / "Figure_3_confusion_matrices_CASAS_SPHERE_upd.png"

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
    print(f"PDF: {out_pdf}")
    print(f"PNG: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
