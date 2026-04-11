# Neuro-Symbolic IoT Pipeline

Reproducible implementation for the paper:

**Building Smarter IoT Systems: A Neuro-Symbolic Approach for Data Federation and Real-Time Processing**

A hybrid AI system combining deep learning perception with OWL/SWRL symbolic reasoning for IoT smart home activity recognition and ambient assisted living (AAL).

![Pipeline Architecture](fig/Figure1_Neuro_Symbolic_IoT_Pipeline.png)

**Figure 1.** End-to-end neuro-symbolic pipeline: raw sensor streams, neural perception, knowledge graph federation, symbolic reasoning, and feedback loop.

---

## Architecture

The pipeline implements three formal algorithms from the paper:

1. **Algorithm 1** — Neural-to-Semantic Federation: maps neural predictions (tensors) to RDF triples in a knowledge graph
2. **Algorithm 2** — Real-Time Neuro-Symbolic Feedback Loop: detects contradictions, retracts false positives, adjusts confidence thresholds
3. **Algorithm 3** — Dynamic Edge-Fog-Cloud Task Orchestration (architecture-level, not in this repo)

### Pipeline Stages

```
[Raw Sensor Data] → [Preprocessing] → [Neural Perception (GRU/LSTM)]
        → [KG Federation (RDF)] → [Symbolic Reasoning (OWL+SWRL+HermiT)]
        → [Feedback Loop (Retraction + Threshold Adaptation)]
```

Each stage is gated by `pipeline.enable_*` flags in config, enabling ablation studies (Table 7).

---

## Project Structure

```
neurosymbolic_iot/
  cli/
    preprocess.py              # Dataset preprocessing CLI
    train_neural.py            # Neural model training CLI
    run_pipeline.py            # Full pipeline CLI (inference → KG → reasoning → feedback)
  data_processing/             # CASAS + SPHERE dataset loaders, windowing, splits
  neural_perception/
    models.py                  # CasasGRUClassifier, SphereLSTMClassifier
    trainer.py                 # Training loop with class-weighted loss
    inference.py               # Softmax confidence extraction, per-sample predictions
    cross_validation.py        # Stratified k-fold CV wrapper
    casas_sequence.py          # CASAS event → token sequence builder
    sphere_sequence.py         # SPHERE acceleration → windowed sequences
  kg_semantic_layer/
    ontology/
      Neuro-Symbolic IoT Smart Home Ontology.ttl  # Production OWL ontology (SOSA/SAREF)
      NeuroSymbolic_IoT_SmartHome_Ontology_SWRL_Rules.ttl  # 11 SWRL rules
      sensor_map.json          # Sensor ID → ontology class mapping (CASAS + SPHERE)
    kg_builder/
      kg_federation_loader.py  # Algorithm 1: predictions → RDF graph
      rdf_writer.py            # Serialize graph + optional GraphDB push
  reasoning_feedback/
    reasoning/
      symbolic_reasoner.py     # owlready2 + HermiT + 11 SWRL rules via Imp() API
      ruleset.swrl             # Human-readable rule reference (not parsed at runtime)
    feedback/
      feedback_loop.py         # Contradiction detection, retrain buffer, feedback cycles
      adapt_kg.py              # KG retraction, confidence threshold adaptation
  utils/
    config.py                  # YAML config loader with inheritance (extends key)
    logging.py                 # Logging setup
    seed.py                    # Global seed (42)

config/
  base.yaml                   # Root config (datasets, neural, kg, reasoning, feedback)
  ns_full.yaml                # Full pipeline (all stages enabled)
  ai_only.yaml                # Neural only (no KG/reasoning/feedback)
  kg_only.yaml                # KG + reasoning (no neural/feedback)
  ns_nofeedback.yaml           # Neural + KG + reasoning (no feedback)
  casas_noise_robustness.yaml  # Noise robustness experiment (adl_error vs adl_noerror)

evaluation/
  run_experiments.py           # Baselines + pipeline/ablation runner (--mode baseline|pipeline|ablation)
  metrics_collector.py         # Pipeline metrics framework (Tables 3-5)
  threshold_sensitivity.py     # Confidence threshold sweep experiment
  federated_reasoning.py       # Cross-dataset federated reasoning experiment
```

---

## Requirements

- **Python** 3.10+
- **Java** (required for the HermiT OWL reasoner used by owlready2)
- **GraphDB** (optional, for persistent triple store)

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch >= 2.0, rdflib >= 7.0, owlready2 >= 0.46, scikit-learn >= 1.2, pandas >= 2.0

---

## Dataset Setup

Place datasets locally — they are gitignored.

### CASAS Kyoto ADL

```
data/raw/casas_kyoto_adl/
  adl_error/*.csv
  adl_noerror/*.csv
```

### SPHERE

```
data/raw/sphere/
  activity.csv
  acceleration_corrected.csv
  pir.csv
  ...
```

---

## Usage

### 1. Preprocess

```bash
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset casas
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset sphere
```

### 2. Train Neural Models

```bash
# CASAS GRU (activity classification)
python -m neurosymbolic_iot.cli.train_neural --config config/base.yaml --dataset casas --task activity

# CASAS GRU (transition detection)
python -m neurosymbolic_iot.cli.train_neural --config config/base.yaml --dataset casas --task transition

# SPHERE LSTM
python -m neurosymbolic_iot.cli.train_neural --config config/base.yaml --dataset sphere
```

### 3. Run Full Pipeline

```bash
python -m neurosymbolic_iot.cli.run_pipeline \
  --config config/ns_full.yaml \
  --dataset casas \
  --task activity \
  --model-dir outputs/neural_perception/<tag>/casas/activity
```

This executes all 4 stages sequentially:

1. **Neural inference** — loads trained model, runs softmax, outputs per-sample predictions
2. **KG construction** — builds RDF graph from predictions (Algorithm 1), serializes to `outputs/kg/<dataset>/populated_kg.ttl`
3. **Symbolic reasoning** — loads ontology + KG into owlready2, defines 11 SWRL rules, runs HermiT
4. **Feedback loop** — detects contradictions, retracts false positives, adjusts thresholds (Algorithm 2)

### 4. Run Ablation Study (Table 7)

```bash
python evaluation/run_experiments.py \
  --config config/ns_full.yaml \
  --mode ablation \
  --datasets casas \
  --model-dir outputs/neural_perception/<tag>/casas/activity
```

Runs all 4 configurations: `ai_only`, `kg_only`, `ns_nofeedback`, `ns_full`.

### 5. Run Baselines

```bash
python evaluation/run_experiments.py --config config/base.yaml --datasets casas,sphere
```

#### PowerShell (default VS Code terminal on Windows):

```bash
$env:PYTHONPATH="."; python evaluation/run_experiments.py --config config/base.yaml --datasets casas,sphere
```

---

## SWRL Rules (11 Rules)

| #  | Category                  | Description                                                        |
| -- | ------------------------- | ------------------------------------------------------------------ |
| 1  | Sensor Grounding          | PIR motion → person location                                      |
| 2  | Sensor Grounding          | Appliance interaction → person location                           |
| 3  | Neuro-Symbolic Validation | High-confidence meal preparation (Kitchen + conf > 0.85)           |
| 4  | Neuro-Symbolic Validation | Posture-based sleeping (Bedroom + Lying + conf > 0.80)             |
| 5  | AAL Anomaly Detection     | Critical fall detection (Bathroom + Lying)                         |
| 6  | AAL Anomaly Detection     | Unattended fire hazard (Kitchen appliance ON + person in Bedroom)  |
| 7  | AAL Anomaly Detection     | Night wandering / sleep disturbance                                |
| 8  | Feedback Trigger          | Spatial hallucination (low confidence + quiet sensors)             |
| 9  | Feedback Trigger          | Contextual hallucination (sleeping in living room + TV ON)         |
| 10 | Feedback Trigger          | Mutually exclusive predictions (PersonalHygiene + MealPreparation) |
| 11 | Feedback Trigger          | Posture-based fall detection (Lying + quiet PIR sensors)           |

Rules are defined programmatically via `owlready2.Imp()` in `symbolic_reasoner.py`. Human-readable reference: `ruleset.swrl`.

---

## Experiment Framework

| Experiment                        | Script / Config                                   | Paper Reference |
| --------------------------------- | ------------------------------------------------- | --------------- |
| 5-fold cross-validation           | `evaluation/run_cv.py`                          | Table 2         |
| Noise robustness (error vs clean) | `evaluation/run_noise_robustness.py`            | Table 3         |
| Confidence threshold sensitivity  | `evaluation/threshold_sensitivity.py`           | Section 4.3     |
| Cross-dataset federated reasoning | `evaluation/federated_reasoning.py`             | Section 3.2.2   |
| Ablation (4 configs)              | `evaluation/run_experiments.py --mode ablation` | Table 7         |
| Pipeline metrics (Tables 4-6)     | `evaluation/metrics_collector.py`               | Tables 4, 5, 6  |

---

## Outputs

```
outputs/
  neural_perception/<tag>/<dataset>/<task>/
    model.pth, metrics.json, label_map.json, vocab.json, confusion_*.csv
  kg/<dataset>/
    populated_kg.ttl
  reasoning/<dataset>/
    reasoning_result.json
  feedback/<dataset>/
    feedback_cycles.json
  pipeline/<dataset>/<tag>/
    pipeline_result.json
```

---

## Configuration

All configs inherit from `config/base.yaml` via the `extends` key. Key sections:

| Section               | Controls                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `datasets`          | Raw data paths, windowing params, split ratios                                                  |
| `neural_perception` | Model hyperparams (GRU/LSTM), training settings                                                 |
| `kg`                | Ontology path, sensor map, GraphDB connection                                                   |
| `reasoning`         | Confidence thresholds (validation: 0.85, anomaly: 0.85, feedback: 0.70)                         |
| `feedback`          | Max cycles (5), retrain buffer size (500), adjustment rate (0.05)                               |
| `pipeline`          | Stage enable flags (`enable_neural`, `enable_kg`, `enable_symbolic`, `enable_feedback`) |

---

## Troubleshooting

| Issue                                  | Fix                                                            |
| -------------------------------------- | -------------------------------------------------------------- |
| `KeyError: 'datasets'`               | Use a config with `datasets` section (e.g., `base.yaml`)   |
| Timezone errors (tz-aware vs tz-naive) | Handled in `train_neural.py` — timestamps normalized to UTC |
| Java not found (HermiT fails)          | Install JRE/JDK; reasoning stage requires Java                 |
| GraphDB push fails                     | Optional — set `kg.graphdb.enabled: false` (default)        |
| Config `inherits` not working        | Use `extends` key (not `inherits`)                         |

---

## Citation

```bibtex
@misc{neurosymbolic_iot_pipeline_repo,
  author       = {Anonymous},
  title        = {Neuro-Symbolic IoT Pipeline: Reproducible Experiments},
  howpublished = {\url{https://github.com/SamRepository/neurosymbolic-iot-pipeline}},
  year         = {2026},
  note         = {Code anonymized for peer-review}
}
```
