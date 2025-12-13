
# Neuro-Symbolic IoT Pipeline (Reproducible Experiments)

Reproducible experimental setup and evaluation code for:

**Building Smarter IoT Systems: A Neuro-Symbolic Approach for Data Federation and Real-Time Processing**

This repository currently provides **Phase 0–1**: environment setup + dataset preprocessing (windowing, basic feature extraction, and splits).

---

## Repository Status

- ✅ Phase 0: project structure + configuration + CLI
- ✅ Phase 1: preprocessing for **CASAS** and **SPHERE**
- ⏳ Next: Phase 2 (neural training + evaluation scripts), Phase 3 (KG building), Phase 4 (symbolic reasoning), Phase 5 (feedback loop)

---

## Requirements

- Python 3.10+ recommended
- Tested on Windows 11 (PowerShell) and should work on Linux/macOS

Install dependencies:

```bash
pip install -r requirements.txt
```


## Dataset Placement

Place datasets locally (do **not** commit them to GitHub):

```
data/
  raw/
    casas/
      p01.csv
      p02.csv
      ...
    sphere/
      activity.csv
      acceleration_corrected.csv
      ...
```

Notes:

* **CASAS** : multiple participant/event CSV files (e.g., `p01.csv ... p26.csv`)
* **SPHERE** : multi-file dataset including `activity.csv` + sensor streams (e.g., `acceleration_corrected.csv`, `pir.csv`, etc.)

---

## Quickstart (Phase 0 + Phase 1)

### 1) Preprocess CASAS

```bash
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset casas
```

### 2) Preprocess SPHERE

```bash
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset sphere
```

---

## Outputs

Preprocessing writes results to:

* `data/processed/casas_windows.parquet` and `data/processed/casas_meta.json`
* `data/processed/sphere_windows.parquet` and `data/processed/sphere_meta.json`

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

## Git / Data Policy

This repo is designed to be reproducible **without** pushing large datasets to GitHub.

Recommended `.gitignore` entries:

```
data/raw/
data/processed/
datasets/
*.zip
*.rar
```

If you need to distribute datasets, use:

* download scripts (preferred), or
* Git LFS, or
* external storage (Zenodo / Drive) with links in the README.

---

## Next Steps (Planned)

Phase 2 will add:

* Training pipelines:
  * GRU-based classifier for CASAS
  * LSTM-based classifier for SPHERE
* Evaluation scripts:
  * accuracy / macro F1 / weighted F1
  * confusion matrices
  * latency profiling hooks

Phase 3+ will add:

* RDF triple generation + KG population
* Symbolic reasoning rules and CEP logic
* Feedback loop for contradiction handling and adaptation

---

## Citation

If you use this repository in academic work, please cite the paper:

**Building Smarter IoT Systems: A Neuro-Symbolic Approach for Data Federation and Real-Time Processing**

```

```
