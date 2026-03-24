# Neuro-Symbolic IoT Pipeline (Reproducible Experiments)

Reproducible experimental setup and evaluation code for the paper:

**Building Smarter IoT Systems: A Neuro-Symbolic Approach for Data Federation and Real-Time Processing**

This repository provides the implementation layers for a hybrid AI system that combines deep learning perception with semantic reasoning.


## Architectural Formalization

The implementation strictly follows the formal algorithms defined in the research paper:
- **Algorithm 1**: Neural-to-Semantic Federation (Maps tensors to RDF triples).
- **Algorithm 2**: Real-Time Neuro-Symbolic Feedback Loop (Handles self-correction).
- **Algorithm 3**: Dynamic Edge-Fog-Cloud Task Orchestration (Manages distributed execution).

---

## Progression Checklist

- ✅ **Phase 0**: Environment setup & project structure.
- ✅ **Phase 1**: Dataset preprocessing (windowing, feature extraction, data splits).
- ✅ **Phase 2**: Neural perception experiments (CASAS GRU models & SPHERE LSTM baselines).
- ✅ **Phase 3**: KG Semantic Layer (Production-ready RDF/OWL Ontology based on SOSA/SAREF).
- ✅ **Phase 4**: Symbolic Reasoning (Curated SWRL Rulebase & logic inference engine).
- ⏳ **Phase 5**: Adaptive Feedback Loop (Integration of automated neural retraining triggers).

---

## Requirements

- **Python**: 3.10+
- **Triple Store**: GraphDB (recommended) or any RDF4J-compatible store for the Semantic Layer.
- **Reasoning**: Java (required for the HermiT reasoner backend used by Owlready2).

Install Python dependencies:
```bash
pip install -r requirements.txt
---

## Dataset Placement

Place datasets locally (do **not** commit them to GitHub).

### Option A — Generic CASAS layout (legacy)

```
data/
  raw/
    casas/
      p01.csv
      p02.csv
      ...
```

### Option B — CASAS Kyoto ADL (errors/noerror) layout (used for Phase 2 CASAS)

This is the dataset layout expected by the `kyoto_adl_errors` loader:

```
data/
  raw/
    casas_kyoto_adl/
      adl_error/
        *.csv
      adl_noerror/
        *.csv
```

### SPHERE (unchanged)

```
data/
  raw/
    sphere/
      activity.csv
      acceleration_corrected.csv
      pir.csv
      ...
```

---

## Configuration

### Recommended: dedicated config for CASAS Kyoto ADL (errors/noerror)

Create (or use) a config file such as `config/casas_kyoto_adl_errors.yaml`:

```yaml
project:
  seed: 42

datasets:
  casas:
    raw_dir: data/raw/casas_kyoto_adl
    format: kyoto_adl_errors
    file_globs:
      - "adl_error/*.csv"
      - "adl_noerror/*.csv"
    window_minutes: 30
    stride_minutes: 5
    min_events_per_window: 1
    label_mode: majority
    feature_mode: event_counts
```

If you keep `config/base.yaml` for other experiments, **do not** point it to the Kyoto ADL path unless you want that dataset to become your default CASAS source.

---

## Phase 1 - Quickstart

### 1) Preprocess CASAS (Kyoto ADL errors/noerror)

```bash
python -m neurosymbolic_iot.cli.preprocess --config config/casas_kyoto_adl_errors.yaml --dataset casas
```

### 2) Preprocess SPHERE

```bash
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset sphere
```

---

## Outputs

Preprocessing writes results to:

- `data/processed/casas_windows.parquet` and `data/processed/casas_meta.json`
- `data/processed/sphere_windows.parquet` and `data/processed/sphere_meta.json`

The `*_windows.parquet` files contain window-level features and labels.
The `*_meta.json` files contain counts and timing information (load, windowing, etc.).

---

## Quick Inspection

### PowerShell (Windows)

```powershell
ls data/processed

python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print('CASAS', df.shape); print(df.columns)"

python -c "import pandas as pd; df=pd.read_parquet('data/processed/sphere_windows.parquet'); print('SPHERE', df.shape); print(df['split'].value_counts()); print(df['label'].value_counts().head(10))"
```

### Bash (Linux/macOS)

```bash
ls data/processed

python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print('CASAS', df.shape); print(df.columns)"

python -c "import pandas as pd; df=pd.read_parquet('data/processed/sphere_windows.parquet'); print('SPHERE', df.shape); print(df['split'].value_counts()); print(df['label'].value_counts().head(10))"
```

---

## Phase 2 — CASAS (Neural Perception)

Phase 2 for CASAS includes:

1) **Tabular baseline** (event-count windows; scikit-learn Logistic Regression)
2) **Sequence model** (GRU over event-token sequences)
3) **Transition task** (binary classifier predicting whether the next window is a *label change*)

### A) Sanity checks we used (CASAS)

These checks help confirm you are using the intended dataset and feature space.

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print('rows', len(df)); print('cols', len(df.columns))"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); ad=[c for c in df.columns if c.startswith('AD1-')]; print('AD1-* columns:', len(ad)); print('Total columns:', len(df.columns))"
```

If you already computed/added `transition` labels in your dataframe:

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print(df['transition'].value_counts()); print('transition_rate =', df['transition'].mean())"
```

### B) Train GRU (CASAS) — transition task (binary)

This uses the `train_neural` CLI and writes a fully self-contained run folder (model + label map + vocab + metrics).

```bash
python -m neurosymbolic_iot.cli.train_neural --config config/casas_kyoto_adl_errors.yaml --dataset casas --task transition
```

Outputs are created under:

```
outputs/
  neural_perception/
    <TAG>/
      casas/
        transition/
          model.pth
          metrics.json
          label_map.json
          vocab.json
          task.json
```

Quickly inspect metrics:

```bash
python -c "import json; from pathlib import Path; p=Path('outputs/neural_perception'); d=sorted(p.glob('*/*/transition/metrics.json'))[-1]; print('latest:', d); m=json.loads(d.read_text()); print('val', m.get('val', {})); print('test', m.get('test', {}))"
```

### C) (Optional) Tabular baseline (LogReg) — reproducible snippet

If you want a minimal “no extra CLI” baseline, this snippet trains a balanced Logistic Regression on event-count features
from the preprocessed CASAS windows file:

```bash
python - <<'PY'
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet("data/processed/casas_windows.parquet").copy()
assert "label" in df.columns, "Missing 'label' column"

# Choose feature columns: event-count features usually look like AD1-*
feat_cols = [c for c in df.columns if c.startswith("AD1-")]
X = df[feat_cols].fillna(0.0).to_numpy()

le = LabelEncoder()
y = le.fit_transform(df["label"].astype(str))

# If preprocessing wrote a 'split' column, use it; else do a simple random split.
if "split" in df.columns:
    train_mask = df["split"].astype(str).str.lower().eq("train")
    val_mask   = df["split"].astype(str).str.lower().eq("val")
    test_mask  = df["split"].astype(str).str.lower().eq("test")
else:
    rng = np.random.RandomState(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_train = int(0.7*n)
    n_val = int(0.15*n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx]=True
    val_mask = np.zeros(n, dtype=bool); val_mask[val_idx]=True
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx]=True

clf = LogisticRegression(
    solver="saga",
    max_iter=5000,
    class_weight="balanced",
    random_state=42,
    n_jobs=None,
)
clf.fit(X[train_mask], y[train_mask])

def report(mask):
    yp = clf.predict(X[mask])
    return {
        "accuracy": float(accuracy_score(y[mask], yp)),
        "f1_macro": float(f1_score(y[mask], yp, average="macro")),
        "f1_weighted": float(f1_score(y[mask], yp, average="weighted")),
    }

val = report(val_mask)
test = report(test_mask)

print("VAL:", val)
print("TEST:", test)
print("classes:", list(le.classes_))
print("confusion_test:\n", confusion_matrix(y[test_mask], clf.predict(X[test_mask])))

Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)
out = {
    "dataset":"casas",
    "seed":42,
    "n_train": int(train_mask.sum()),
    "n_val": int(val_mask.sum()),
    "n_test": int(test_mask.sum()),
    "n_features": int(len(feat_cols)),
    "n_classes": int(len(le.classes_)),
    "classes": list(le.classes_),
    "model":"LogisticRegression",
    "val": val,
    "test": test,
}
with open("outputs/evaluation/casas_logreg_metrics.json","w",encoding="utf-8") as f:
    json.dump(out,f,indent=2)
print("Wrote outputs/evaluation/casas_logreg_metrics.json")
PY
```

---

## Troubleshooting Notes (what we hit during Phase 2 CASAS)

### 1) `KeyError: 'datasets'` when running `train_neural`

Cause: you passed a config that did not contain a `datasets:` section (e.g., a neural-only config).

Fix: run with `config/base.yaml` (or a dataset config like `config/casas_kyoto_adl_errors.yaml`) that includes `datasets.casas`.

### 2) Timezone errors (tz-aware vs tz-naive) during sequence building

Symptoms included:

- `TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type`
- `TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp`

Fix: normalize timestamps to **UTC-naive** (convert with `utc=True` then drop tz). This is handled in the current `train_neural.py`
implementation; if you reintroduce custom timestamp parsing, keep it consistent.

### 3) Warning: “no grouping columns found” for transition computation

If the windows dataframe has no `source_file/session_id/participant/...` fields, transitions are computed globally (mixing independent timelines).

Recommended improvement: propagate `source_file` (or a similar identifier) into the window dataframe inside preprocessing so transitions are computed per file/session.

---

## Git / Data Policy

This repo is designed to be reproducible **without** pushing large datasets to GitHub.

Recommended `.gitignore` entries:

```
data/raw/
data/processed/
datasets/
outputs/
*.zip
*.rar
```

If you need to distribute datasets, use:

- download scripts (preferred), or
- Git LFS, or
- external storage (Zenodo / Drive) with links in the README.

---

## Phase 3 — KG Semantic Layer (Knowledge Graph)

This phase implements the **Semantic Federation** module. It transforms neural outputs and sensor metadata into a unified Knowledge Graph using industry standards.

### Key Features:
- **Standards-Compliant**: Extends the `W3C SOSA/SSN` and `ETSI SAREF` ontologies.
- **Neuro-Symbolic Bridge**: Implements "Reification" to store neural embeddings (`hasLatentVector`) and confidence scores directly in the graph.
- **Format**: Serialized in **Turtle (.ttl)** for maximum interoperability.

### Artifacts:
- `neurosymbolic_iot/kg_semantic_layer/ontology/Neuro-Symbolic_IoT_SmartHome.ttl`: The core production-ready ontology.

---

## Phase 4 — Symbolic Reasoning Engine

This phase implements the **Symbolic Models (Reasoning)** module. It applies high-level logic to validate neural predictions and detect complex behavioral anomalies.

### Logical Foundation:
- **Rule Language**: SWRL (Semantic Web Rule Language).
- **Reasoning Engine**: Python bridge via `Owlready2` and the `HermiT` reasoner.
- **Curated Rulebase**: includes rules for:
    - **Spatial Grounding**: Mapping PIR and SmartPlug states to person location.
    - **AAL Hazard Detection**: Detecting falls in bathrooms or unattended fire hazards.
    - **Feedback Triggers**: Identifying contradictions between AI predictions and physical constraints.

### Running Inference:
```bash
# Core reasoning script (under active development)
python -m neurosymbolic_iot.cli.reasoning --ontology data/ontology.ttl --rules data/rules.ttl
```
---

## Troubleshooting & Notes

- **Timezone Handling**: All timestamps are normalized to **UTC-naive** during preprocessing to ensure compatibility across different sensor sources.
- **Memory Management**: For large Knowledge Graph updates, ensure GraphDB is allocated at least 4GB of heap memory.
- **Anonymization**: This repository has been anonymized for the double-blind peer-review process.

---

## Citation

To cite this repository in your research:

```bibtex
@misc{neurosymbolic_iot_pipeline_repo,
  author       = {Anonymous},
  title        = {Neuro-Symbolic IoT Pipeline: Reproducible Experiments},
  howpublished = {\url{https://github.com/SamRepository/neurosymbolic-iot-pipeline}},
  year         = {2026},
  note         = {Code anonymized for peer-review}
}
```