# The 11-rule base, as executed (E6 — reviewer items R1-12, R2-3)

These are the rules that actually fire in every reported experiment: the SPARQL
CONSTRUCT translations in `rule_executor.RULES`, selected by the project's
`rule_executor: python` setting. owlready2's bundled HermiT/Pellet perform OWL-DL
classification only and do not execute SWRL rule bodies, so the SWRL forms in
`symbolic_reasoner.define_swrl_rules` are never evaluated at run time.

| # | Category | Rule | Precondition (SPARQL WHERE) | Consequent (CONSTRUCT) | Feedback? |
|---|---|---|---|---|---|
| 1 | sensor grounding | PIR motion implies person location | `?p a nsiot:Person . ?s a nsiot:PIRMotionSensor . ?s nsiot:isLocatedIn ?room . ?s nsiot:hasState nsiot:StateMotionDetected .` | `?p nsiot:isLocatedIn ?room` | no |
| 2 | sensor grounding | Appliance interaction implies person location | `?p a nsiot:Person . ?plug a nsiot:SmartAppliancePlug . ?plug nsiot:isLocatedIn ?room . ?plug nsiot:hasState nsiot:StateON .` | `?p nsiot:isLocatedIn ?room` | no |
| 3 | validation | High-confidence Cook in Kitchen validation | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:predictsActivity nsiot:MealPreparation . ?pred nsiot:hasConfidenceScore ?conf . FILTER(?conf  0.85) ?p nsiot:isLocatedIn ?room . ?room a nsiot:Kitchen .` | `?p nsiot:performsActivity nsiot:MealPreparation . ?e a nsiot:ValidatedEvent .` | no |
| 4 | validation | High-confidence Eat in Kitchen validation | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:predictsActivity nsiot:Eating . ?pred nsiot:hasConfidenceScore ?conf . FILTER(?conf  0.80) ?p nsiot:isLocatedIn ?room . ?room a nsiot:Kitchen .` | `?p nsiot:performsActivity nsiot:Eating . ?e a nsiot:ValidatedEvent .` | no |
| 5 | AAL anomaly | Cook without kitchen appliance activation | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:predictsActivity nsiot:MealPreparation . ?pred nsiot:hasConfidenceScore ?conf . FILTER(?conf  0.85) FILTER NOT EXISTS { ?plug a nsiot:SmartAppliancePlug . ?plug nsiot:isLocatedIn ?k . ?k a nsiot:Kitch…` | `?e a nsiot:CriticalAnomaly . ?e nsiot:hasAlertType nsiot:UnattendedAppliance .` | no |
| 6 | AAL anomaly | Unattended fire hazard (kitchen plug ON, person elsewhere) | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?plug a nsiot:SmartAppliancePlug . ?plug nsiot:isLocatedIn ?k . ?k a nsiot:Kitchen . ?plug nsiot:hasState nsiot:StateON . ?p nsiot:isLocatedIn ?other . FILTER(?other != ?k)` | `?e a nsiot:CriticalAnomaly . ?e nsiot:hasAlertType nsiot:UnattendedFireHazard .` | no |
| 7 | AAL anomaly | Nocturnal kitchen activity | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:hasTimeContext ?ctx . ?ctx a nsiot:NightTimeContext . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:predictsActivity ?act . FILTER(?act IN (nsiot:MealPreparation, nsiot:Eating, nsiot:Housekeeping)) ?pred nsiot:hasConfidenceScore ?conf . …` | `?e a nsiot:BehavioralAnomaly . ?e nsiot:hasAlertType nsiot:NocturnalActivity .` | no |
| 8 | feedback trigger | Spatial hallucination (kitchen activity without kitchen motion) | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:predictsActivity ?act . FILTER(?act IN (nsiot:MealPreparation, nsiot:Eating, nsiot:Housekeeping)) ?pred nsiot:hasConfidenceScore ?conf . FILTER(?conf < 0.80) FILTER NOT EXISTS { ?ks a nsiot:PIRMotionS…` | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:FalsePositiveHallucination .` | yes |
| 9 | feedback trigger | Contextual mismatch (Cook predicted, person not in Kitchen) | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:predictsActivity nsiot:MealPreparation . ?pred nsiot:hasConfidenceScore ?conf . FILTER(?conf < 0.85) ?p nsiot:isLocatedIn ?room . FILTER NOT EXISTS { ?p nsiot:isLocatedIn ?k . ?k a nsiot:Kitchen .` | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:ContextualMismatch .` | yes |
| 10 | feedback trigger | Low-margin AND low-confidence top-2 disagreement | `?e a nsiot:Event . ?e nsiot:isBasedOnPrediction ?pred1 . ?pred1 nsiot:isAlternativePrediction false . ?pred1 nsiot:predictsActivity ?act1 . ?pred1 nsiot:hasConfidenceScore ?conf1 . ?e nsiot:isBasedOnPrediction ?pred2 . ?pred2 nsiot:isAlternativePrediction true . ?pred2 nsiot:predictsActivity ?act…` | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:MutuallyExclusiveActivities .` | yes |
| 11 | feedback trigger | Unsupported claim (low conf + no sensor evidence) | `?e a nsiot:Event . ?e nsiot:involvesPerson ?p . ?e nsiot:isBasedOnPrediction ?pred . ?pred nsiot:hasConfidenceScore ?conf . FILTER(?conf < 0.75) FILTER NOT EXISTS { ?p nsiot:isLocatedIn ?_room .` | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:UnsupportedClaim .` | yes |

## Disjointness of feedback

For the executed rule set the manuscript's taxonomy holds: **True** — rules 8–11 are exactly the rules emitting `FeedbackRequired`, and no rule asserts a ground-truth label (labels come from dataset annotations only).

## ⚠ The SWRL definitions are not the executed definitions

`symbolic_reasoner.define_swrl_rules` defines a second, materially different set of eleven rules. Only **3 of 11** are equivalent to their executed counterparts: 4 assert different consequents and 8 fire on different evidence.

Critically, rule(s) [11] differ in *category*: the SWRL form emits a `CriticalAnomaly` while the executed form emits `FeedbackRequired`. The 8–11 disjointness claim is therefore **False** for the SWRL set. An appendix that transcribes the SWRL rules would document a system that never ran and would break the disjointness argument the manuscript relies on.

| # | Executed consequent | SWRL consequent | Status |
|---|---|---|---|
| 1 | `?p nsiot:isLocatedIn ?room` | `nsiot:isLocatedIn(?p, ?room)` | equivalent |
| 2 | `?p nsiot:isLocatedIn ?room` | `nsiot:isLocatedIn(?p, ?room)` | equivalent |
| 3 | `?p nsiot:performsActivity nsiot:MealPreparation . ?e a nsiot:Validated` | `nsiot:performsActivity(?p, nsiot:MealPreparation), nsiot:ValidatedEven` | equivalent |
| 4 | `?p nsiot:performsActivity nsiot:Eating . ?e a nsiot:ValidatedEvent .` | `nsiot:performsActivity(?p, nsiot:Sleeping), nsiot:ValidatedEvent(?e)` | **DIVERGENT (trigger and consequent)** |
| 5 | `?e a nsiot:CriticalAnomaly . ?e nsiot:hasAlertType nsiot:UnattendedApp` | `nsiot:CriticalAnomaly(?e), nsiot:hasAlertType(?e, nsiot:FallAlert)` | **DIVERGENT (trigger and consequent)** |
| 6 | `?e a nsiot:CriticalAnomaly . ?e nsiot:hasAlertType nsiot:UnattendedFir` | `nsiot:CriticalAnomaly(?e), nsiot:hasAlertType(?e, nsiot:UnattendedFire` | **DIVERGENT (trigger only — same consequent, different evidence)** |
| 7 | `?e a nsiot:BehavioralAnomaly . ?e nsiot:hasAlertType nsiot:NocturnalAc` | `nsiot:BehavioralAnomaly(?e), nsiot:hasAlertType(?e, nsiot:SleepDisturb` | **DIVERGENT (trigger and consequent)** |
| 8 | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:FalsePositiv` | `nsiot:FeedbackRequired(?e), nsiot:hasErrorType(?e, nsiot:FalsePositive` | **DIVERGENT (trigger only — same consequent, different evidence)** |
| 9 | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:ContextualMi` | `nsiot:FeedbackRequired(?e), nsiot:hasErrorType(?e, nsiot:ContextualMis` | **DIVERGENT (trigger only — same consequent, different evidence)** |
| 10 | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:MutuallyExcl` | `nsiot:FeedbackRequired(?e), nsiot:hasErrorType(?e, nsiot:MutuallyExclu` | **DIVERGENT (trigger only — same consequent, different evidence)** |
| 11 | `?e a nsiot:FeedbackRequired . ?e nsiot:hasErrorType nsiot:UnsupportedC` | `nsiot:CriticalAnomaly(?e), nsiot:hasAlertType(?e, nsiot:FallAlert)` | **DIVERGENT (trigger and consequent)** |

Consequents are compared by the ontology classes they assert. Preconditions are written in different languages (SWRL atoms vs SPARQL triple patterns) and cannot be diffed syntactically, so they are compared by the set of nsiot: terms each body references — enough to establish whether two rules fire on the same evidence.

## Same rule set across datasets?

ONE rule set, hard-coded, instantiated for CASAS Kyoto ADL vocabulary; it is applied unchanged to every dataset. rule_executor.RULES is a single module-level list and rule_executor.fire_rules takes no dataset argument and performs no dataset branching - the same eleven SPARQL CONSTRUCT queries run against whatever graph is passed. There are no per-dataset rule variants anywhere in the codebase.

The consequence must be stated plainly, because 'one rule set for all datasets' sounds like a generality claim and here it is closer to the opposite. The rule bodies are written against CASAS Kyoto ADL's class vocabulary (MealPreparation, Eating, Housekeeping, Kitchen, SmartAppliancePlug), as the module's own comment records. On any dataset whose vocabulary differs, rules referencing absent classes are structurally incapable of firing rather than being merely inapplicable to the data. This is measurable: on CASAS Aruba, five of the eleven rules (2, 6, 8, 9, 11) never fired across all five folds, and rule 10 alone accounted for roughly three quarters of all firings (see ablation_diagnostics.json). SPHERE, whose vocabulary is postures and PIR rooms rather than kitchen appliances, would exercise even fewer.

The defensible statement for the manuscript is therefore: a single fixed rule base is shared across datasets, authored against the CASAS Kyoto ADL vocabulary, with per-dataset coverage varying widely and reported per dataset - not that the rule base generalises across the benchmarks.
