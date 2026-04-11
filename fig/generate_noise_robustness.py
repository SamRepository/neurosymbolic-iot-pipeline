"""Generate noise robustness bar chart and LaTeX table."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    results_path = Path("outputs/noise_robustness/noise_robustness_results.json")
    data = json.loads(results_path.read_text())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
    })

    subsets = ["adl_noerror", "adl_error", "combined"]
    labels = ["No Error\n(clean)", "With Error\n(noisy)", "Combined"]
    metrics = ["accuracy_mean", "f1_macro_mean", "f1_weighted_mean"]
    metric_labels = ["Accuracy", "F1-macro", "F1-weighted"]
    std_keys = ["accuracy_std", "f1_macro_std", "f1_weighted_std"]
    colors = ["#309CF5", "#F9A629", "#DF53F8"]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(metrics))
    width = 0.25

    for i, (subset, label, color) in enumerate(zip(subsets, labels, colors)):
        means = [data[subset][m] for m in metrics]
        stds = [data[subset][s] for s in std_keys]
        bars = ax.bar(x + i * width, means, width, label=label,
                      color=color, alpha=0.85, yerr=stds,
                      capsize=4, error_kw={"linewidth": 1.2})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    """ax.set_title("Noise Robustness: Error vs. No-Error Sensor Streams",
                 fontsize=12, fontweight="bold", pad=12)"""
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Delta annotation
    d_acc = data["adl_error"]["accuracy_mean"] - data["adl_noerror"]["accuracy_mean"]
    ax.annotate(f"{d_acc:+.1%}", xy=(0 + width, data["adl_error"]["accuracy_mean"]),
                xytext=(0 + width, 0.45), fontsize=8, ha="center", color="#D32F2F",
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2))

    out_dir = Path("fig")
    fig.savefig(out_dir / "Figure4_noise_robustness_comparison.pdf",
                format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "Figure4_noise_robustness_comparison.png",
                format="png", bbox_inches="tight", dpi=300)
    print(f"Saved: {out_dir / 'Figure4_noise_robustness_comparison.pdf'}")
    plt.close(fig)

    # Print LaTeX table
    print()
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Noise robustness: GRU activity recognition under clean vs.\ error-injected sensor streams (5-fold CV).}")
    print(r"    \label{tab:noise-robustness}")
    print(r"    \begin{tabular}{lcccc}")
    print(r"        \toprule")
    print(r"        \textbf{Condition} & \textbf{N} & \textbf{Accuracy} & \textbf{F1-macro} & \textbf{F1-weighted} \\")
    print(r"        \midrule")
    for name, display in [("adl_noerror", "No Error (clean)"), ("adl_error", "With Error (noisy)"), ("combined", "Combined")]:
        s = data[name]
        print(f"        {display} & {s['n_windows']} & "
              f"${s['accuracy_mean']:.2%} \pm {s['accuracy_std']:.2%}$ & "
              f"${s['f1_macro_mean']:.2%} \pm {s['f1_macro_std']:.2%}$ & "
              f"${s['f1_weighted_mean']:.2%} \pm {s['f1_weighted_std']:.2%}$ \\\\")
    print(r"        \midrule")
    e = data["adl_error"]
    n = data["adl_noerror"]
    d_a = e["accuracy_mean"] - n["accuracy_mean"]
    d_m = e["f1_macro_mean"] - n["f1_macro_mean"]
    d_w = e["f1_weighted_mean"] - n["f1_weighted_mean"]
    print(f"        $\Delta$ (Error $-$ Clean) & --- & ${d_a:+.2%}$ & ${d_m:+.2%}$ & ${d_w:+.2%}$ \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
