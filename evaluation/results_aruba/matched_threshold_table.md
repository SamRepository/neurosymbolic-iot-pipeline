# Matched-threshold ablation table (E2 rebuttal artifact)

Derived from the published Aruba run (commit `f5c6cd3`) via a reconstruction that
reproduces its per-fold AI-Only metrics to 1e-9. **No re-run was required.**

Because the symbolic layer changes zero labels (see ablation_diagnostics.json), AI-Only, NeSy-NoFeedback and NeSy-Full are numerically identical whenever they are scored at the same confidence threshold. The lift reported in the published Table 11 arises solely because NeSy-Full is scored at 0.75 while AI-Only is scored at 0.70. No re-run is required to establish this: the values below are derived from the same validated reconstruction that reproduces the published AI-Only numbers to 1e-9.

## All configurations scored at confidence threshold 0.70

| Configuration | F1-w (%) | Correctness (%) | FP (%) | Coverage (%) |
|---|---|---|---|---|
| AI-Only | 83.4 ± 7.0 | 82.6 ± 6.8 | 17.4 ± 6.8 | 56.1 |
| NeSy-NoFeedback | 83.4 ± 7.0 | 82.6 ± 6.8 | 17.4 ± 6.8 | 56.1 |
| NeSy-Full | 83.4 ± 7.0 | 82.6 ± 6.8 | 17.4 ± 6.8 | 56.1 |
| KG+Rules † | N/A | 0.0 | — | — |

*The three neural-containing configurations are numerically identical at a
matched threshold — the symbolic layer changes no labels.*

† KG+Rules is threshold-independent (it applies no confidence gate). Its
correctness is defined in only 3 of 5 folds and rests on 1–21 rule-asserted
events; the 0.0 is that degenerate measurement, not a 0 % accuracy result on
a full evaluation set. Coverage is not comparable for a rule-only baseline.

**Coverage is not 100 %.** Both thresholds answer only part of the held-out
set — the published F1 figures describe the answered subset, and the
manuscript should say so alongside them.

## All configurations scored at confidence threshold 0.75

| Configuration | F1-w (%) | Correctness (%) | FP (%) | Coverage (%) |
|---|---|---|---|---|
| AI-Only | 85.0 ± 6.4 | 84.6 ± 6.2 | 15.4 ± 6.2 | 50.5 |
| NeSy-NoFeedback | 85.0 ± 6.4 | 84.6 ± 6.2 | 15.4 ± 6.2 | 50.5 |
| NeSy-Full | 85.0 ± 6.4 | 84.6 ± 6.2 | 15.4 ± 6.2 | 50.5 |
| KG+Rules † | N/A | 0.0 | — | — |

*The three neural-containing configurations are numerically identical at a
matched threshold — the symbolic layer changes no labels.*

† KG+Rules is threshold-independent (it applies no confidence gate). Its
correctness is defined in only 3 of 5 folds and rests on 1–21 rule-asserted
events; the 0.0 is that degenerate measurement, not a 0 % accuracy result on
a full evaluation set. Coverage is not comparable for a rule-only baseline.

**Coverage is not 100 %.** Both thresholds answer only part of the held-out
set — the published F1 figures describe the answered subset, and the
manuscript should say so alongside them.

## What the feedback loop actually does: adaptive abstention

| Gate | Coverage (%) | F1-w (%) |
|---|---|---|
| 0.70 (before feedback) | 56.1 | 83.4 |
| 0.75 (after one cycle) | 50.5 | 85.0 |

The loop abstains on 5.6 pp more windows (per-fold: [17, 21, 23, 23, 16]) and gains 1.6 pp F1 on those it still answers.
