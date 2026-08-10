# Per-dataset and knowledge-graph statistics (E6 — reviewer item R2-3)

| Dataset | Windows (experiment) | Windows (processed) | Classes (CV) | Classes (processed) | Train | Test | KG triples | Predicates | nsiot entities |
|---|---|---|---|---|---|---|---|---|---|
| casas_kyoto | 400 | 176 | 5 | 4 | 320 | 80 | 1666 | 26 | 401 |
| casas_aruba | 1792 | 10771 | 10 | 11 | 1434 | 358 | 6568 | 27 | 1591 |
| sphere | — | 139 | — | 7 | 97 | 21 | — | — | — |

Classes are reported on both counting bases because they disagree; see `class_count_discrepancy` in the JSON for the per-dataset label differences.
- `casas_kyoto`: CV corpus 5 classes vs processed store 4; CV-only ['WashHands'], processed-only —.
- `casas_aruba`: CV corpus 10 classes vs processed store 11; CV-only —, processed-only ['Work'].

KG statistics are measured on fold 0's serialized graph, after rule firing.

**Two window counts are reported deliberately.** The CV scripts rebuild windows
from the raw stream using the experiment config rather than reading
`data/processed`, so the two figures differ. Quote whichever matches the
experiment being described; do not mix them.

- `casas_kyoto`: processed parquet 176 windows vs CV run 400 windows.

- `casas_aruba`: processed parquet 10771 windows vs CV run 1792 windows.

**SPHERE is markedly data-limited** (reviewer item R1-6): 139 windows total with
a 97/21/21 train/val/test split, i.e. 21 test windows across 20 label columns.
Any generalisation claim resting on SPHERE should be tempered accordingly.

## Ontology

| File | Triples | OWL classes | Object properties | Datatype properties |
|---|---|---|---|---|
| Neuro-Symbolic IoT Smart Home Ontology.ttl | 142 | 33 | 12 | 4 |
| NeuroSymbolic_IoT_SmartHome_Ontology_SWRL_Rules.ttl | not rdflib-parseable (owlready2 SWRL syntax) | — | — | — |
| NeuroSymbolicOntology_basic.ttl | 29 | 10 | 3 | 1 |

## Sensor and activity vocabulary per dataset

### casas_kyoto

- Sensor types: nsiot:DoorContactSensor, nsiot:PIRMotionSensor, nsiot:SmartAppliancePlug, nsiot:TemperatureSensor, nsiot:WaterFlowSensor
- Mapped rooms: 4
- Activity map entries: 5
- Class labels: Clean, Cook, Eat, PhoneCall, WashHands

### casas_aruba

- Sensor types: nsiot:DoorContactSensor, nsiot:PIRMotionSensor, nsiot:SmartAppliancePlug, nsiot:TemperatureSensor, nsiot:WaterFlowSensor
- Mapped rooms: 4
- Activity map entries: 5
- Class labels: Eating, EnterHome, Housekeeping, LeaveHome, MealPreparation, Medication, PersonalHygiene, PhoneCall, Relax, Sleeping

### sphere

- Sensor types: nsiot:PIRMotionSensor
- Mapped rooms: 5
- Activity map entries: 20
