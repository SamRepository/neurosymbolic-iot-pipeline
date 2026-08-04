# Over-correction and false-retraction safety metrics (E4 — reviewer item R1-10)

Source: evaluation/results_aruba/ablation_cv.json (commit f5c6cd3)

## Headline

64 of 166 feedback flags (38.6 %) were raised on windows the network had classified correctly. This is the false-flag rate the manuscript should report beside the false-positive-reduction figure.

| Pathway | Denominator | Over-corrections | Rate |
|---|---|---|---|
| Flagged though correct (rules 8–11) | 166 flags | 64 | **38.6 %** |
| Retracted though correct (rule 8) | 0 eligible | — | undefined — mechanism never activated |
| Correct predictions abstained on | 100 abstained | 65 | **65.0 %** |
| AAL alerts on a false premise (rules 5–7) | 24 alerts | 10 | **41.7 %** |

## By rule

| Rule | Category | Events | Over-corrections | Rate |
|---|---|---|---|---|
| R10 | feedback_trigger | 166 flags | 64 on correct | 38.6 % |
| R5 | aal_anomaly | 20 alerts | 6 false premise | 30.0 % |
| R7 | aal_anomaly | 4 alerts | 4 false premise | 100.0 % |

## Retraction

No triples were retracted anywhere in the run. adapt_kg.retract_false_positives acts only on FalsePositiveHallucination flags, which only rule 8 emits, and rule 8 never fired on Aruba (0 eligible flags). The false-retraction rate is therefore undefined rather than zero, and the manuscript must not report it as evidence that retraction is safe: the mechanism was never exercised on this dataset.

## Abstention

The over-correction that did occur is abstention. Raising the gate dropped 100 windows across the five folds, of which 65 (65.0 %) carried correct predictions. Those are valid behaviours the system silently stopped reporting - the closest analogue in this pipeline to the reviewer's 'valid atypical behaviour erroneously retracted', and the number that matters for an AAL deployment, since an unanswered window raises no alert at all.

## AAL alerts

Of 24 AAL alerts raised by rules 5-7, 10 (41.7 %) were raised on windows whose underlying activity classification was wrong, so the alert rests on a false premise. This is a lower bound on the alert false-alarm rate, not the rate itself.

## What this data cannot establish

CASAS Aruba provides activity labels, not anomaly labels. An alert raised on a correctly classified window therefore cannot be adjudicated as a true or false anomaly from the data - only alerts built on a misclassification can be identified with certainty. Any fuller safety claim needs a dataset with annotated anomalies or expert adjudication, and the manuscript should say so rather than implying the false-alarm rate is fully characterised.
