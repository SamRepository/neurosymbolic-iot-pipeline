"""
Ablation statistics and test justification (E3 - FGCS revision)
===============================================================
Addresses reviewer item R1-7: report the per-fold scores, justify the
one-sided hypothesis choice, and make the practical magnitude and the
statistical fragility visible together.

Reads the published per-fold values from ``ablation_cv.json`` (it reports the
numbers the manuscript quotes, it does not recompute them) and emits:

  * a per-fold table of F1-w / correctness / FP rate for all four configurations;
  * one-sided AND two-sided paired Wilcoxon p-values side by side, plus a
    sign test and a paired t-test as distribution-free / parametric checks;
  * 95 % CIs (Student t, df = 4, t = 2.776) on every mean and on every paired
    mean difference;
  * an explicit attainability analysis: at n = 5 the Wilcoxon null distribution
    is coarse, so p = 0.03125 is the smallest one-sided p-value reachable at
    all and corresponds to "every fold moved the same way", nothing finer.

Where ``matched_threshold_table.json`` (E2b) is present, its result is carried
into the output so the fragility discussion and the threshold confound are
reported in one place rather than in two documents that must be cross-read.

Usage:
  PYTHONPATH=. python evaluation/run_ablation_statistics.py --results-dir evaluation/results_aruba
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scistats

log = logging.getLogger(__name__)

CONFIG_NAMES = ["AI-Only", "KG+Rules", "NeSy-NoFeedback", "NeSy-Full"]
METRICS = ["F1_weighted", "Correctness", "FP_rate"]

# Direction in which NeSy-Full is hypothesised to move relative to a baseline.
METRIC_DIRECTION = {
    "F1_weighted": "greater",
    "Correctness": "greater",
    "FP_rate": "less",
}

T_CRIT_DF4 = 2.776  # t_{0.975, 4}


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def _describe(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Mean, sd and 95 % CI for a per-fold metric vector (n = 5)."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "sd": None, "ci95": None, "ci95_half": None,
                "n": 0, "per_fold": []}
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if n > 1:
        sd = float(arr.std(ddof=1))
        t_crit = float(scistats.t.ppf(0.975, n - 1))
        half = t_crit * sd / math.sqrt(n)
    else:
        sd, half = 0.0, 0.0
    return {
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "ci95": [round(mean - half, 6), round(mean + half, 6)],
        "ci95_half": round(half, 6),
        "n": n,
        "per_fold": [round(v, 6) for v in vals],
    }


# ---------------------------------------------------------------------------
# Wilcoxon attainability at small n
# ---------------------------------------------------------------------------

def _wilcoxon_null_distribution(n: int) -> Dict[int, float]:
    """Exact null distribution of W+ for the signed-rank test with n non-zero ranks.

    Every one of the 2^n sign assignments is equally likely under H0, so the
    distribution is obtained by direct enumeration. Used to show what p-values
    are reachable at all when n = 5.
    """
    counts: Dict[int, int] = {}
    ranks = range(1, n + 1)
    for signs in itertools.product([0, 1], repeat=n):
        w_plus = sum(r for r, s in zip(ranks, signs) if s)
        counts[w_plus] = counts.get(w_plus, 0) + 1
    total = 2 ** n
    return {w: c / total for w, c in sorted(counts.items())}


def _attainability(n: int) -> Dict[str, Any]:
    """What the test can and cannot resolve at this sample size."""
    dist = _wilcoxon_null_distribution(n)
    w_max = n * (n + 1) // 2
    p_min_one_sided = dist[w_max]
    return {
        "n_folds": n,
        "W_max": w_max,
        "min_attainable_p_one_sided": round(p_min_one_sided, 6),
        "min_attainable_p_two_sided": round(min(1.0, 2 * p_min_one_sided), 6),
        "interpretation": (
            f"With n = {n} paired folds the Wilcoxon null distribution is coarse: the smallest "
            f"one-sided p-value attainable at all is {p_min_one_sided:.5f} = 1/{2**n}, reached "
            f"only when W = W_max = {w_max}, i.e. when every fold moves in the hypothesised "
            f"direction. A reported p of {p_min_one_sided:.4f} therefore carries exactly one bit "
            f"of information — 'all {n} folds agreed in sign' — and cannot distinguish a large "
            f"effect from a marginal one. The corresponding two-sided p-value, "
            f"{min(1.0, 2*p_min_one_sided):.4f}, does not reach alpha = 0.05 at any effect size."
        ),
    }


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

def _paired_comparison(
    baseline: Sequence[Optional[float]],
    target: Sequence[Optional[float]],
    direction: str,
) -> Dict[str, Any]:
    """Wilcoxon (one- and two-sided), sign test, paired t-test, and effect size."""
    pairs = [(float(b), float(t)) for b, t in zip(baseline, target)
             if b is not None and t is not None]
    if len(pairs) < 2:
        return {"note": "insufficient paired observations", "n_paired": len(pairs)}

    base_arr = np.asarray([b for b, _ in pairs], dtype=float)
    tgt_arr = np.asarray([t for _, t in pairs], dtype=float)
    # Improvement is positive for 'greater' metrics and negative for 'less'.
    diffs = tgt_arr - base_arr
    improvement = diffs if direction == "greater" else -diffs
    n = len(improvement)

    n_positive = int(np.sum(improvement > 0))
    n_negative = int(np.sum(improvement < 0))
    n_zero = int(np.sum(improvement == 0))

    result: Dict[str, Any] = {
        "n_paired": n,
        "direction_tested": direction,
        "mean_difference": round(float(diffs.mean()), 6),
        "mean_improvement": round(float(improvement.mean()), 6),
        "per_fold_difference": [round(float(d), 6) for d in diffs],
        "folds_favouring_target": n_positive,
        "folds_favouring_baseline": n_negative,
        "folds_tied": n_zero,
    }

    sd_diff = float(improvement.std(ddof=1)) if n > 1 else 0.0
    result["cohens_d_paired"] = round(float(improvement.mean() / sd_diff), 4) if sd_diff > 0 else 0.0
    if sd_diff > 0:
        half = float(scistats.t.ppf(0.975, n - 1)) * sd_diff / math.sqrt(n)
        md = float(diffs.mean())
        result["mean_difference_ci95"] = [round(md - half, 6), round(md + half, 6)]
    else:
        result["mean_difference_ci95"] = [result["mean_difference"], result["mean_difference"]]

    if np.all(improvement == 0):
        result["wilcoxon"] = {
            "W": None, "p_one_sided": None, "p_two_sided": None,
            "note": (
                "All per-fold differences are exactly zero: the signed-rank statistic has no "
                "non-zero ranks to sum and the test is undefined. The configurations are "
                "numerically identical, not merely statistically indistinguishable."
            ),
        }
        result["sign_test_p_two_sided"] = None
        result["paired_t_test"] = {"t": None, "p_two_sided": None}
        return result

    # Wilcoxon: 'greater' on the improvement vector tests improvement > 0.
    zeros = np.zeros_like(improvement)
    one_sided = scistats.wilcoxon(improvement, zeros, alternative="greater", zero_method="wilcox")
    two_sided = scistats.wilcoxon(improvement, zeros, alternative="two-sided", zero_method="wilcox")
    result["wilcoxon"] = {
        "W": float(one_sided.statistic),
        "p_one_sided": round(float(one_sided.pvalue), 6),
        "p_two_sided": round(float(two_sided.pvalue), 6),
        "significant_one_sided_alpha_0.05": bool(one_sided.pvalue < 0.05),
        "significant_two_sided_alpha_0.05": bool(two_sided.pvalue < 0.05),
    }

    n_nonzero = n_positive + n_negative
    if n_nonzero > 0:
        result["sign_test_p_two_sided"] = round(
            float(scistats.binomtest(n_positive, n_nonzero, 0.5).pvalue), 6
        )
    else:
        result["sign_test_p_two_sided"] = None

    t_stat = scistats.ttest_rel(tgt_arr, base_arr)
    result["paired_t_test"] = {
        "t": round(float(t_stat.statistic), 4),
        "p_two_sided": round(float(t_stat.pvalue), 6),
    }
    return result


ONE_SIDED_JUSTIFICATION = (
    "The one-sided formulation is the pre-registered engineering hypothesis: the ablation asks "
    "whether adding a component to the pipeline improves the metric, and a significant result in "
    "the opposite direction (the added component making the system worse) would lead to the same "
    "action as no result at all - the component would not be adopted. That is the standard "
    "condition under which a one-sided test is defensible. Two caveats must nevertheless be "
    "reported alongside it, and this artifact reports both. First, the two-sided p-value is given "
    "for every comparison so the reader can apply the stricter criterion directly; where the "
    "one-sided test is significant and the two-sided test is not, that is stated rather than "
    "omitted. Second, at n = 5 the choice of tail is decisive rather than cosmetic: the smallest "
    "attainable one-sided p is 0.03125 and the smallest attainable two-sided p is 0.0625, so no "
    "five-fold comparison of any effect size can reach alpha = 0.05 two-sided. The one-sided "
    "result should therefore be read as a directional consistency statement (all folds agreed), "
    "not as strong evidence of a large effect."
)


def _fragility_assessment(comparisons: Dict[str, Any], attainability: Dict[str, Any]) -> Dict[str, Any]:
    """State precisely what is and is not fragile — the two are easy to conflate.

    The reviewer asked for the "statistically significant" framing to be moderated. The
    data support a more specific statement than "the result is fragile": the paired
    difference is consistent and precisely estimated, but the non-parametric test cannot
    resolve it at n = 5, and the difference measures a gate change rather than the
    architecture (E2). Reporting only the two-sided Wilcoxon p would understate the
    measurement and misplace the real caveat.
    """
    f1 = comparisons["F1_weighted__AI-Only_vs_NeSy-Full"]
    ci = f1["mean_difference_ci95"]
    return {
        "what_is_robust": {
            "statement": (
                "The paired difference is consistent and precisely estimated. All five folds move "
                "in the same direction, the per-fold F1 differences span a narrow band "
                f"({min(f1['per_fold_difference'])*100:.2f} to {max(f1['per_fold_difference'])*100:.2f} pp), "
                f"the paired t-test gives p = {f1['paired_t_test']['p_two_sided']:.4f} two-sided, and the 95 % CI on "
                f"the mean difference [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp excludes zero. The effect is not noise."
            ),
            "paired_t_p_two_sided": f1["paired_t_test"]["p_two_sided"],
            "mean_difference_ci95_pp": [round(ci[0] * 100, 3), round(ci[1] * 100, 3)],
        },
        "what_is_fragile": {
            "statement": (
                "The non-parametric evidence is weak in a way that is structural, not incidental. "
                f"The two-sided Wilcoxon p is {f1['wilcoxon']['p_two_sided']:.4f} and the sign test gives "
                f"{f1['sign_test_p_two_sided']:.4f}; at n = 5 neither can fall below "
                f"{attainability['min_attainable_p_two_sided']:.4f} regardless of effect size. Any claim of "
                "two-sided non-parametric significance is unreachable with five folds, so the "
                "'statistically significant' phrasing should not rest on the Wilcoxon result alone."
            ),
            "wilcoxon_p_two_sided": f1["wilcoxon"]["p_two_sided"],
            "sign_test_p_two_sided": f1["sign_test_p_two_sided"],
        },
        "the_decisive_caveat": (
            "Neither of the above is the main problem. The difference is real and well measured, "
            "but E2 established that it is produced entirely by the feedback cycle raising the "
            "confidence gate from 0.70 to 0.75: at a matched gate the configurations are "
            "numerically identical. The correct moderation is therefore not 'the effect may be "
            "noise' - it is 'the effect is real but is a coverage/accuracy trade-off from adaptive "
            "abstention, not evidence that symbolic reasoning corrects classifications'. Softening "
            "the claim on statistical grounds alone would misdescribe the finding."
        ),
        "recommended_phrasing": (
            "Across five stratified folds NeSy-Full improves weighted F1 over AI-Only by "
            f"{f1['mean_difference']*100:.2f} pp (95 % CI [{ci[0]*100:.2f}, {ci[1]*100:.2f}]; all five folds "
            f"positive; one-sided Wilcoxon W = {f1['wilcoxon']['W']:.0f}, p = {f1['wilcoxon']['p_one_sided']:.4f}, the "
            "smallest value attainable at n = 5; two-sided p = "
            f"{f1['wilcoxon']['p_two_sided']:.4f}). This gain reflects adaptive abstention: the feedback cycle "
            "raises the confidence gate, reducing answered-window coverage while increasing "
            "accuracy on the windows that remain."
        ),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}"


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    desc = payload["descriptive"]
    per_fold = payload["per_fold_table"]
    k = payload["_meta"]["k"]

    lines.append("# Per-fold ablation statistics (E3 — reviewer item R1-7)")
    lines.append("")
    lines.append(f"Source: `{payload['_meta']['source']}` — the values the manuscript quotes,")
    lines.append("reported per fold rather than recomputed.")
    lines.append("")

    lines.append("## Per-fold scores, all configurations")
    lines.append("")
    header = "| Configuration | Metric | " + " | ".join(f"Fold {i}" for i in range(k)) + " | Mean ± SD | 95 % CI |"
    lines.append(header)
    lines.append("|" + "---|" * (k + 4))
    for cfg in CONFIG_NAMES:
        for metric in METRICS:
            vals = per_fold[cfg][metric]
            d = desc[cfg][metric]
            cells = " | ".join(_fmt_pct(v) for v in vals)
            if d["mean"] is None:
                summary = "N/A | N/A"
            else:
                summary = (
                    f"{_fmt_pct(d['mean'])} ± {_fmt_pct(d['sd'])} | "
                    f"[{_fmt_pct(d['ci95'][0])}, {_fmt_pct(d['ci95'][1])}]"
                )
            label = metric.replace("_", "-").replace("F1-weighted", "F1-w")
            lines.append(f"| {cfg} | {label} | {cells} | {summary} |")
    lines.append("")
    lines.append("All values are percentages. KG+Rules has no defined weighted F1 (the rule-only")
    lines.append("baseline produces predictions on a sparse validated subset only).")
    lines.append("")

    lines.append("## Paired tests — one-sided and two-sided side by side")
    lines.append("")
    lines.append("| Comparison | Metric | Mean Δ | 95 % CI on Δ | W | p (1-sided) | p (2-sided) | Sign test | Cohen's d | Folds |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, comp in payload["paired_comparisons"].items():
        if "wilcoxon" not in comp:
            continue
        baseline, target = key.split("__")[1].split("_vs_")
        metric = key.split("__")[0].replace("F1_weighted", "F1-w").replace("_", " ")
        w = comp["wilcoxon"]
        w_str = "—" if w["W"] is None else f"{w['W']:.0f}"
        p1 = "undefined" if w["p_one_sided"] is None else f"{w['p_one_sided']:.4f}"
        p2 = "undefined" if w["p_two_sided"] is None else f"{w['p_two_sided']:.4f}"
        sign = "—" if comp["sign_test_p_two_sided"] is None else f"{comp['sign_test_p_two_sided']:.4f}"
        ci = comp["mean_difference_ci95"]
        lines.append(
            f"| {baseline} → {target} | {metric} | {comp['mean_difference']*100:+.2f} pp "
            f"| [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | {w_str} | {p1} | {p2} | {sign} "
            f"| {comp['cohens_d_paired']:+.2f} "
            f"| {comp['folds_favouring_target']}/{comp['n_paired']} |"
        )
    lines.append("")

    att = payload["attainability"]
    lines.append("## Why the sample size limits what these p-values can say")
    lines.append("")
    lines.append(att["interpretation"])
    lines.append("")
    lines.append("| n | W_max | Min attainable p (1-sided) | Min attainable p (2-sided) |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {att['n_folds']} | {att['W_max']} | {att['min_attainable_p_one_sided']:.5f} "
        f"| {att['min_attainable_p_two_sided']:.5f} |"
    )
    lines.append("")

    lines.append("## Justification of the one-sided choice")
    lines.append("")
    lines.append(payload["one_sided_justification"])
    lines.append("")

    frag = payload["fragility_assessment"]
    lines.append("## What is robust and what is fragile — they are not the same thing")
    lines.append("")
    lines.append(f"**Robust.** {frag['what_is_robust']['statement']}")
    lines.append("")
    lines.append(f"**Fragile.** {frag['what_is_fragile']['statement']}")
    lines.append("")
    lines.append(f"**The decisive caveat.** {frag['the_decisive_caveat']}")
    lines.append("")
    lines.append("### Suggested phrasing for the manuscript")
    lines.append("")
    lines.append(f"> {frag['recommended_phrasing']}")
    lines.append("")

    confound = payload.get("threshold_confound")
    if confound:
        lines.append("## Necessary caveat: the compared configurations use different gates")
        lines.append("")
        lines.append(confound["summary"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3: per-fold ablation statistics and test justification.")
    p.add_argument("--results-dir", default="evaluation/results_aruba",
                   help="Directory holding ablation_cv.json")
    p.add_argument("--out", default=None,
                   help="Output JSON (default: <results-dir>/ablation_statistics.json)")
    return p.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from neurosymbolic_iot.utils.logging import setup_logging

    args = parse_args()
    setup_logging("INFO")

    results_dir = Path(args.results_dir)
    source = results_dir / "ablation_cv.json"
    stored = json.loads(source.read_text(encoding="utf-8"))
    per_fold = stored["per_fold"]
    k = int(stored["metadata"]["k"])

    descriptive = {
        cfg: {metric: _describe(per_fold[cfg][metric]) for metric in METRICS}
        for cfg in CONFIG_NAMES
    }

    comparisons: Dict[str, Any] = {}
    for baseline in ("AI-Only", "NeSy-NoFeedback"):
        for metric in METRICS:
            key = f"{metric}__{baseline}_vs_NeSy-Full"
            comparisons[key] = _paired_comparison(
                per_fold[baseline][metric],
                per_fold["NeSy-Full"][metric],
                METRIC_DIRECTION[metric],
            )

    attainability = _attainability(k)

    # Carry the E2b matched-threshold finding in, if it has been generated.
    confound: Optional[Dict[str, Any]] = None
    matched_path = results_dir / "matched_threshold_table.json"
    if matched_path.exists():
        matched = json.loads(matched_path.read_text(encoding="utf-8"))
        tradeoff = matched["adaptive_abstention_tradeoff"]
        confound = {
            "summary": (
                "The published comparison is not a like-for-like architecture contrast. AI-Only is "
                "scored at a confidence gate of 0.70 and NeSy-Full at 0.75, because the feedback "
                "cycle raises the gate; the two configurations therefore answer different numbers "
                "of windows (coverage "
                f"{tradeoff['coverage@0.70']['mean']*100:.1f} % vs "
                f"{tradeoff['coverage@0.75']['mean']*100:.1f} %). Scored at a matched gate the two "
                "are numerically identical and the paired test is undefined (all five differences "
                "exactly zero). Whatever significance framing the manuscript adopts, the magnitude "
                "reported here is attributable to the gate change rather than to symbolic "
                "correction — see ablation_diagnostics.json and matched_threshold_table.json."
            ),
            "coverage_at_0.70": tradeoff["coverage@0.70"]["mean"],
            "coverage_at_0.75": tradeoff["coverage@0.75"]["mean"],
            "matched_threshold_test": "undefined (all per-fold differences exactly zero)",
            "source": str(matched_path).replace("\\", "/"),
        }
    else:
        log.warning("matched_threshold_table.json not found — threshold confound not cross-linked")

    payload = {
        "_meta": {
            "script": "evaluation/run_ablation_statistics.py",
            "commit": _git_commit(),
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source": str(source).replace("\\", "/"),
            "k": k,
            "seed": stored["metadata"].get("seed"),
            "note": (
                "Reports the published per-fold values; it does not recompute them. "
                "Statistics are computed from those values exactly as stored."
            ),
        },
        "per_fold_table": {
            cfg: {metric: per_fold[cfg][metric] for metric in METRICS} for cfg in CONFIG_NAMES
        },
        "descriptive": descriptive,
        "paired_comparisons": comparisons,
        "attainability": attainability,
        "fragility_assessment": _fragility_assessment(comparisons, attainability),
        "one_sided_justification": ONE_SIDED_JUSTIFICATION,
        "threshold_confound": confound,
        "published_test_reproduced": {
            "F1_weighted__AI-Only_vs_NeSy-Full": {
                "published": {"W": 15.0, "p_one_sided": 0.0312, "cohens_d": 2.48},
                "recomputed": {
                    "W": comparisons["F1_weighted__AI-Only_vs_NeSy-Full"]["wilcoxon"]["W"],
                    "p_one_sided": comparisons["F1_weighted__AI-Only_vs_NeSy-Full"]["wilcoxon"]["p_one_sided"],
                    "cohens_d": comparisons["F1_weighted__AI-Only_vs_NeSy-Full"]["cohens_d_paired"],
                },
            }
        },
    }

    out_path = Path(args.out) if args.out else results_dir / "ablation_statistics.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    # ----- console summary -----
    print()
    print("=" * 92)
    print(f"  E3 — PER-FOLD STATISTICS AND TEST JUSTIFICATION (k = {k}, seed {payload['_meta']['seed']})")
    print("=" * 92)
    print("  Per-fold F1-w (%):")
    for cfg in CONFIG_NAMES:
        vals = per_fold[cfg]["F1_weighted"]
        d = descriptive[cfg]["F1_weighted"]
        cells = "  ".join(_fmt_pct(v).rjust(5) for v in vals)
        mean_str = "N/A" if d["mean"] is None else (
            f"{_fmt_pct(d['mean'])} ± {_fmt_pct(d['sd'])}  CI [{_fmt_pct(d['ci95'][0])}, {_fmt_pct(d['ci95'][1])}]"
        )
        print(f"    {cfg:<17} {cells}   |  {mean_str}")
    print()
    print("  Paired tests (NeSy-Full vs baseline):")
    print(f"    {'comparison':<44} {'W':>4} {'p-1sided':>10} {'p-2sided':>10} {'d':>7} {'folds':>7}")
    for key, comp in comparisons.items():
        if "wilcoxon" not in comp:
            continue
        w = comp["wilcoxon"]
        w_str = "—" if w["W"] is None else f"{w['W']:.0f}"
        p1 = "undef" if w["p_one_sided"] is None else f"{w['p_one_sided']:.4f}"
        p2 = "undef" if w["p_two_sided"] is None else f"{w['p_two_sided']:.4f}"
        flag = "" if (w["p_two_sided"] is None or w["p_two_sided"] < 0.05) else "  <- NOT sig. 2-sided"
        print(f"    {key:<44} {w_str:>4} {p1:>10} {p2:>10} {comp['cohens_d_paired']:>+7.2f} "
              f"{comp['folds_favouring_target']}/{comp['n_paired']:<5}{flag}")
    print()
    att = attainability
    print(f"  Attainability at n = {att['n_folds']}: min p one-sided {att['min_attainable_p_one_sided']:.5f}, "
          f"two-sided {att['min_attainable_p_two_sided']:.5f}")
    print("  => no 5-fold comparison can reach alpha = 0.05 two-sided, at any effect size.")
    if confound:
        print(f"  Threshold confound: coverage {confound['coverage_at_0.70']*100:.1f} % (AI-Only@0.70) "
              f"vs {confound['coverage_at_0.75']*100:.1f} % (NeSy-Full@0.75); matched-gate test undefined.")
    print("=" * 92)

    log.info("Wrote %s and %s", out_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
