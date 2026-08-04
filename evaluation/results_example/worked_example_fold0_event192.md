# Worked end-to-end example (E10 — reviewer item R2-8)

A single real window from the published Aruba run: fold 0, event index 192 (`nsiot:event_192`).

## Stage 1 — Raw sensor events

Window `2012-08-22T23:08:54.512364+00:00` → `2012-08-22T23:38:54.512364+00:00` (118 events). Annotated activity: **Relax**.

| t+s | Sensor | State |
|---|---|---|
| 5 | `LivingRoom` | ON |
| 6 | `LivingRoom` | OFF |
| 251 | `LivingRoom` | ON |
| 252 | `LivingRoom` | OFF |
| 252 | `LivingRoom` | ON |
| 254 | `LivingRoom` | OFF |
| 335 | `LivingRoom` | ON |
| 336 | `LivingRoom` | OFF |
| 339 | `LivingRoom` | ON |
| 340 | `LivingRoom` | OFF |
| 521 | `LivingRoom` | ON |
| 522 | `LivingRoom` | OFF |
| 700 | `LivingRoom` | ON |
| 701 | `LivingRoom` | OFF |
| 703 | `LivingRoom` | ON |
| 704 | `LivingRoom` | OFF |
| 737 | `LivingRoom` | ON |
| 739 | `LivingRoom` | OFF |
| 740 | `LivingRoom` | ON |
| 741 | `LivingRoom` | OFF |
| 747 | `LivingRoom` | ON |
| 748 | `LivingRoom` | OFF |
| 762 | `LivingRoom` | ON |
| 764 | `LivingRoom` | OFF |
| 766 | `LivingRoom` | ON |
| … | *(93 further events)* | |

## Stage 2 — Neural perception

- Model: `unknown` (CASAS GRU)
- Top-1: **MealPreparation** at confidence **0.8624**
- Top-2: Housekeeping at 0.1155 (margin 0.7469)

## Stage 3 — Triples emitted into the knowledge graph

24 triples describe this window:

```turtle
nsiot:pred_192  nsiot:generatedAt  2012-08-22T23:08:54.512364+00:00 .
nsiot:pred_192  nsiot:generatedByModel  unknown .
nsiot:pred_192  nsiot:hasConfidenceScore  0.8623695969581604 .
nsiot:pred_192  nsiot:isAlternativePrediction  false .
nsiot:pred_192  nsiot:predictsActivity  MealPreparation .
nsiot:pred_192  nsiot:type  NeuralPrediction .
nsiot:pred_192_alt  nsiot:hasConfidenceScore  0.11548091471195221 .
nsiot:pred_192_alt  nsiot:isAlternativePrediction  true .
nsiot:pred_192_alt  nsiot:predictsActivity  Housekeeping .
nsiot:pred_192_alt  nsiot:type  NeuralPrediction .
nsiot:event_192  nsiot:hasAlertType  NocturnalActivity .
nsiot:event_192  nsiot:hasAlertType  UnattendedAppliance .
nsiot:event_192  nsiot:hasTemporalEntity  time_192 .
nsiot:event_192  nsiot:hasTimeContext  nightctx_192 .
nsiot:event_192  nsiot:involvesPerson  Participant1 .
nsiot:event_192  nsiot:isBasedOnPrediction  pred_192 .
nsiot:event_192  nsiot:isBasedOnPrediction  pred_192_alt .
nsiot:event_192  nsiot:type  BehavioralAnomaly .
nsiot:event_192  nsiot:type  CriticalAnomaly .
nsiot:event_192  nsiot:type  Event .
nsiot:event_192  nsiot:type  ValidatedEvent .
nsiot:time_192  nsiot:inXSDDateTimeStamp  2012-08-22T23:08:54.512364+00:00 .
nsiot:time_192  nsiot:type  Instant .
nsiot:nightctx_192  nsiot:type  NightTimeContext .
```

## Stage 4 — Rule firings

Each of the eleven rule bodies was executed against the graph; a rule is listed as
fired only if it yields a triple whose subject is this event.

| Rule | Category | Fired here? | Asserted |
|---|---|---|---|
| R1 | sensor grounding | no | — |
| R2 | sensor grounding | no | — |
| R3 | validation | **yes** | `type` → `ValidatedEvent` |
| R4 | validation | no | — |
| R5 | AAL anomaly | **yes** | `type` → `CriticalAnomaly`; `hasAlertType` → `UnattendedAppliance` |
| R6 | AAL anomaly | no | — |
| R7 | AAL anomaly | **yes** | `hasAlertType` → `NocturnalActivity`; `type` → `BehavioralAnomaly` |
| R8 | feedback trigger | no | — |
| R9 | feedback trigger | no | — |
| R10 | feedback trigger | no | — |
| R11 | feedback trigger | no | — |

### Supporting evidence (why those rules matched)

Sensors active in this window: `Kitchen`, `LivingRoom`

- `Participant1` `isLocatedIn` `Bathroom`
- `Participant1` `isLocatedIn` `Bedroom`
- `Participant1` `isLocatedIn` `Kitchen`
- `Participant1` `isLocatedIn` `LivingRoom`
- `Participant1` `performsActivity` `Eating`
- `Participant1` `performsActivity` `MealPreparation`
- `Participant1` `type` `Person`

*The KG asserts sensor state and location globally rather than linking sensors to the event that observed them: there is no event->sensor edge. The sensor facts above are therefore matched by sensor id from the window's raw tokens, not followed from the event node. This is the same gap that forced E1 to add explicit event->room provenance triples before cross-source location conflicts could be scoped to events.*

## Stage 5 — Final decision and explanation trace

- **Decision:** MealPreparation (ground truth Relax — INCORRECT)
- **Confidence** 0.8624; answered at the 0.7 gate: yes; at the post-feedback 0.75 gate: yes
- **Validated by:** [3]
- **Alerts raised:** R5, R7
- **Feedback flags:** none
- **Label changed by the symbolic layer:** no

*The symbolic layer never rewrites the argmax label (E2): validation annotates, and feedback flags are applied only to predictions already below the confidence gate. The final label is the network's top-1.*
