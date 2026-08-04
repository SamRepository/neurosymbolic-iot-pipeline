# Worked end-to-end example (E10 — reviewer item R2-8)

A single real window from the published Aruba run: fold 0, event index 295 (`nsiot:event_295`).

## Stage 1 — Raw sensor events

Window `2012-09-09T09:08:54.512364+00:00` → `2012-09-09T09:38:54.512364+00:00` (256 events). Annotated activity: **MealPreparation**.

| t+s | Sensor | State |
|---|---|---|
| 17 | `Kitchen` | ON |
| 18 | `Kitchen` | OFF |
| 19 | `Kitchen` | ON |
| 21 | `Kitchen` | OFF |
| 25 | `Kitchen` | ON |
| 28 | `Kitchen` | ON |
| 29 | `Kitchen` | OFF |
| 30 | `Kitchen` | ON |
| 30 | `Kitchen` | ON |
| 31 | `Kitchen` | OFF |
| 32 | `Kitchen` | OFF |
| 33 | `Kitchen` | ON |
| 33 | `Kitchen` | ON |
| 34 | `Kitchen` | OFF |
| 35 | `Kitchen` | OFF |
| 36 | `Kitchen` | ON |
| 36 | `Kitchen` | ON |
| 36 | `Kitchen` | OFF |
| 37 | `Kitchen` | OFF |
| 37 | `Kitchen` | ON |
| 38 | `Kitchen` | OFF |
| 39 | `Kitchen` | OFF |
| 41 | `Kitchen` | ON |
| 44 | `Kitchen` | OFF |
| 46 | `Kitchen` | ON |
| … | *(231 further events)* | |

## Stage 2 — Neural perception

- Model: `unknown` (CASAS GRU)
- Top-1: **MealPreparation** at confidence **0.9357**
- Top-2: Housekeeping at 0.0513 (margin 0.8844)

## Stage 3 — Triples emitted into the knowledge graph

20 triples describe this window:

```turtle
nsiot:pred_295  nsiot:generatedAt  2012-09-09T09:08:54.512364+00:00 .
nsiot:pred_295  nsiot:generatedByModel  unknown .
nsiot:pred_295  nsiot:hasConfidenceScore  0.9357162117958069 .
nsiot:pred_295  nsiot:isAlternativePrediction  false .
nsiot:pred_295  nsiot:predictsActivity  MealPreparation .
nsiot:pred_295  nsiot:type  NeuralPrediction .
nsiot:pred_295_alt  nsiot:hasConfidenceScore  0.05134272947907448 .
nsiot:pred_295_alt  nsiot:isAlternativePrediction  true .
nsiot:pred_295_alt  nsiot:predictsActivity  Housekeeping .
nsiot:pred_295_alt  nsiot:type  NeuralPrediction .
nsiot:event_295  nsiot:hasAlertType  UnattendedAppliance .
nsiot:event_295  nsiot:hasTemporalEntity  time_295 .
nsiot:event_295  nsiot:involvesPerson  Participant1 .
nsiot:event_295  nsiot:isBasedOnPrediction  pred_295 .
nsiot:event_295  nsiot:isBasedOnPrediction  pred_295_alt .
nsiot:event_295  nsiot:type  CriticalAnomaly .
nsiot:event_295  nsiot:type  Event .
nsiot:event_295  nsiot:type  ValidatedEvent .
nsiot:time_295  nsiot:inXSDDateTimeStamp  2012-09-09T09:08:54.512364+00:00 .
nsiot:time_295  nsiot:type  Instant .
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
| R5 | AAL anomaly | **yes** | `hasAlertType` → `UnattendedAppliance`; `type` → `CriticalAnomaly` |
| R6 | AAL anomaly | no | — |
| R7 | AAL anomaly | no | — |
| R8 | feedback trigger | no | — |
| R9 | feedback trigger | no | — |
| R10 | feedback trigger | no | — |
| R11 | feedback trigger | no | — |

### Supporting evidence (why those rules matched)

Sensors active in this window: `Bathroom`, `DiningRoom`, `Kitchen`, `LivingRoom`, `OutsideDoor`

- `Participant1` `isLocatedIn` `Bathroom`
- `Participant1` `isLocatedIn` `Bedroom`
- `Participant1` `isLocatedIn` `Kitchen`
- `Participant1` `isLocatedIn` `LivingRoom`
- `Participant1` `performsActivity` `Eating`
- `Participant1` `performsActivity` `MealPreparation`
- `Participant1` `type` `Person`

*The KG asserts sensor state and location globally rather than linking sensors to the event that observed them: there is no event->sensor edge. The sensor facts above are therefore matched by sensor id from the window's raw tokens, not followed from the event node. This is the same gap that forced E1 to add explicit event->room provenance triples before cross-source location conflicts could be scoped to events.*

## Stage 5 — Final decision and explanation trace

- **Decision:** MealPreparation (ground truth MealPreparation — correct)
- **Confidence** 0.9357; answered at the 0.7 gate: yes; at the post-feedback 0.75 gate: yes
- **Validated by:** [3]
- **Alerts raised:** R5
- **Feedback flags:** none
- **Label changed by the symbolic layer:** no

*The symbolic layer never rewrites the argmax label (E2): validation annotates, and feedback flags are applied only to predictions already below the confidence gate. The final label is the network's top-1.*
