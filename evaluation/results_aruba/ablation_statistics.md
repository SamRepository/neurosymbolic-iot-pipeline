# Per-fold ablation statistics (E3 — reviewer item R1-7)

Source: `evaluation/results_aruba/ablation_cv.json` — the values the manuscript quotes,
reported per fold rather than recomputed.

## Per-fold scores, all configurations

| Configuration | Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD | 95 % CI |
|---|---|---|---|---|---|---|---|---|
| AI-Only | F1-w | 86.7 | 72.3 | 81.4 | 90.4 | 86.1 | 83.4 ± 7.0 | [74.7, 92.1] |
| AI-Only | Correctness | 85.7 | 72.4 | 80.0 | 90.3 | 84.7 | 82.6 ± 6.8 | [74.2, 91.0] |
| AI-Only | FP-rate | 14.3 | 27.6 | 20.0 | 9.7 | 15.3 | 17.4 ± 6.8 | [9.0, 25.8] |
| KG+Rules | F1-w | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| KG+Rules | Correctness | 0.0 | 0.0 | 0.0 | N/A | N/A | 0.0 ± 0.0 | [0.0, 0.0] |
| KG+Rules | FP-rate | 100.0 | 100.0 | 100.0 | N/A | N/A | 100.0 ± 0.0 | [100.0, 100.0] |
| NeSy-NoFeedback | F1-w | 86.7 | 72.3 | 81.4 | 90.4 | 86.1 | 83.4 ± 7.0 | [74.7, 92.1] |
| NeSy-NoFeedback | Correctness | 85.7 | 72.4 | 80.0 | 90.3 | 84.7 | 82.6 ± 6.8 | [74.2, 91.0] |
| NeSy-NoFeedback | FP-rate | 14.3 | 27.6 | 20.0 | 9.7 | 15.3 | 17.4 ± 6.8 | [9.0, 25.8] |
| NeSy-Full | F1-w | 87.8 | 74.8 | 83.4 | 91.5 | 87.4 | 85.0 ± 6.4 | [77.1, 92.9] |
| NeSy-Full | Correctness | 86.8 | 75.6 | 82.3 | 92.1 | 86.4 | 84.6 ± 6.2 | [77.0, 92.3] |
| NeSy-Full | FP-rate | 13.2 | 24.4 | 17.7 | 7.9 | 13.6 | 15.4 ± 6.2 | [7.7, 23.0] |

All values are percentages. KG+Rules has no defined weighted F1 (the rule-only
baseline produces predictions on a sparse validated subset only).

## Paired tests — one-sided and two-sided side by side

| Comparison | Metric | Mean Δ | 95 % CI on Δ | W | p (1-sided) | p (2-sided) | Sign test | Cohen's d | Folds |
|---|---|---|---|---|---|---|---|---|---|
| AI-Only → NeSy-Full | F1-w | +1.61 pp | [+0.80, +2.42] | 15 | 0.0312 | 0.0625 | 0.0625 | +2.48 | 5/5 |
| AI-Only → NeSy-Full | Correctness | +2.03 pp | [+1.11, +2.94] | 15 | 0.0312 | 0.0625 | 0.0625 | +2.75 | 5/5 |
| AI-Only → NeSy-Full | FP rate | -2.03 pp | [-2.94, -1.11] | 15 | 0.0312 | 0.0625 | 0.0625 | +2.75 | 5/5 |
| NeSy-NoFeedback → NeSy-Full | F1-w | +1.61 pp | [+0.80, +2.42] | 15 | 0.0312 | 0.0625 | 0.0625 | +2.48 | 5/5 |
| NeSy-NoFeedback → NeSy-Full | Correctness | +2.03 pp | [+1.11, +2.94] | 15 | 0.0312 | 0.0625 | 0.0625 | +2.75 | 5/5 |
| NeSy-NoFeedback → NeSy-Full | FP rate | -2.03 pp | [-2.94, -1.11] | 15 | 0.0312 | 0.0625 | 0.0625 | +2.75 | 5/5 |

## Why the sample size limits what these p-values can say

With n = 5 paired folds the Wilcoxon null distribution is coarse: the smallest one-sided p-value attainable at all is 0.03125 = 1/32, reached only when W = W_max = 15, i.e. when every fold moves in the hypothesised direction. A reported p of 0.0312 therefore carries exactly one bit of information — 'all 5 folds agreed in sign' — and cannot distinguish a large effect from a marginal one. The corresponding two-sided p-value, 0.0625, does not reach alpha = 0.05 at any effect size.

| n | W_max | Min attainable p (1-sided) | Min attainable p (2-sided) |
|---|---|---|---|
| 5 | 15 | 0.03125 | 0.06250 |

## Justification of the one-sided choice

The one-sided formulation is the pre-registered engineering hypothesis: the ablation asks whether adding a component to the pipeline improves the metric, and a significant result in the opposite direction (the added component making the system worse) would lead to the same action as no result at all - the component would not be adopted. That is the standard condition under which a one-sided test is defensible. Two caveats must nevertheless be reported alongside it, and this artifact reports both. First, the two-sided p-value is given for every comparison so the reader can apply the stricter criterion directly; where the one-sided test is significant and the two-sided test is not, that is stated rather than omitted. Second, at n = 5 the choice of tail is decisive rather than cosmetic: the smallest attainable one-sided p is 0.03125 and the smallest attainable two-sided p is 0.0625, so no five-fold comparison of any effect size can reach alpha = 0.05 two-sided. The one-sided result should therefore be read as a directional consistency statement (all folds agreed), not as strong evidence of a large effect.

## What is robust and what is fragile — they are not the same thing

**Robust.** The paired difference is consistent and precisely estimated. All five folds move in the same direction, the per-fold F1 differences span a narrow band (1.09 to 2.58 pp), the paired t-test gives p = 0.0052 two-sided, and the 95 % CI on the mean difference [+0.80, +2.42] pp excludes zero. The effect is not noise.

**Fragile.** The non-parametric evidence is weak in a way that is structural, not incidental. The two-sided Wilcoxon p is 0.0625 and the sign test gives 0.0625; at n = 5 neither can fall below 0.0625 regardless of effect size. Any claim of two-sided non-parametric significance is unreachable with five folds, so the 'statistically significant' phrasing should not rest on the Wilcoxon result alone.

**The decisive caveat.** Neither of the above is the main problem. The difference is real and well measured, but E2 established that it is produced entirely by the feedback cycle raising the confidence gate from 0.70 to 0.75: at a matched gate the configurations are numerically identical. The correct moderation is therefore not 'the effect may be noise' - it is 'the effect is real but is a coverage/accuracy trade-off from adaptive abstention, not evidence that symbolic reasoning corrects classifications'. Softening the claim on statistical grounds alone would misdescribe the finding.

### Suggested phrasing for the manuscript

> Across five stratified folds NeSy-Full improves weighted F1 over AI-Only by 1.61 pp (95 % CI [0.80, 2.42]; all five folds positive; one-sided Wilcoxon W = 15, p = 0.0312, the smallest value attainable at n = 5; two-sided p = 0.0625). This gain reflects adaptive abstention: the feedback cycle raises the confidence gate, reducing answered-window coverage while increasing accuracy on the windows that remain.

## Necessary caveat: the compared configurations use different gates

The published comparison is not a like-for-like architecture contrast. AI-Only is scored at a confidence gate of 0.70 and NeSy-Full at 0.75, because the feedback cycle raises the gate; the two configurations therefore answer different numbers of windows (coverage 56.1 % vs 50.5 %). Scored at a matched gate the two are numerically identical and the paired test is undefined (all five differences exactly zero). Whatever significance framing the manuscript adopts, the magnitude reported here is attributable to the gate change rather than to symbolic correction — see ablation_diagnostics.json and matched_threshold_table.json.
