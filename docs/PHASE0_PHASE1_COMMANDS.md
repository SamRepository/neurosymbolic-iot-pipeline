# Phase 0 + Phase 1 — Reproducible Command Log (Windows 11 / PowerShell)

This section documents the exact commands to reproduce Phase 0 (environment + repo readiness) and Phase 1 (CASAS + SPHERE preprocessing).

---

## Phase 0 — Environment + Repository Setup

### 0.1) Open the repository folder

**Purpose:** move into the project root (where `config/`, `neurosymbolic_iot/`, `README.md` exist).

```powershell
cd "C:\path\to\neurosymbolic-iot-pipeline"
```

### 0.2) (Optional) Confirm Python + pip

**Purpose:** ensure Python is available and check versions.

```powershell
python --version
pip --version
```

### 0.3) Create a virtual environment (recommended)

**Purpose:** isolate dependencies for reproducibility.

```powershell
python -m venv .venv
```

### 0.4) Activate the virtual environment

**Purpose:** use the isolated Python environment.

```powershell
.\.venv\Scripts\Activate.ps1
```

### 0.5) Install dependencies

**Purpose:** install all required libraries (pandas, sklearn, pyarrow, tqdm, etc.).

```powershell
pip install -r requirements.txt
```

### 0.6) Check that your repo does NOT track datasets / outputs

**Purpose:** avoid GitHub push failures (100MB limit) and keep repo clean.

```powershell
git status
```

---

## Phase 1 — Dataset Preparation + Preprocessing

### 1.1) Create dataset folders (if missing)

**Purpose:** standardize the dataset layout expected by the pipeline.

```powershell
mkdir data\raw\casas -Force
mkdir data\raw\sphere -Force
```

### 1.2) Place dataset files in the correct locations (manual step)

**Purpose:** the preprocessing scripts read data from these paths.

* Put CASAS files under:
  * `data/raw/casas/p01.csv ... p26.csv`
* Put SPHERE files under:
  * `data/raw/sphere/activity.csv`
  * `data/raw/sphere/acceleration_corrected.csv` (or alternative streams)
  * other SPHERE CSVs as needed

Verify files exist:

```powershell
ls data\raw\casas
ls data\raw\sphere
```

### 1.3) Run CASAS preprocessing

**Purpose:** parse CASAS event logs → create sliding windows + labels → save parquet + meta JSON.

```powershell
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset casas
```

**Expected outputs:**

* `data/processed/casas_windows.parquet`
* `data/processed/casas_meta.json`

### 1.4) Run SPHERE preprocessing

**Purpose:** load SPHERE activity labels + sensor stream → align labels to sensor timeline → windowing + splits → save parquet + meta JSON.

```powershell
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset sphere
```

**Expected outputs:**

* `data/processed/sphere_windows.parquet`
* `data/processed/sphere_meta.json`

### 1.5) Verify that outputs were generated

**Purpose:** confirm Phase 1 produced the expected artifacts.

```powershell
ls data\processed
```

### 1.6) (Optional) Quick summary (shapes only)

**Purpose:** quickly confirm both parquet files exist and show their dimensions.

```powershell
python -c "import pandas as pd; print('CASAS', pd.read_parquet('data/processed/casas_windows.parquet').shape); print('SPHERE', pd.read_parquet('data/processed/sphere_windows.parquet').shape)"
```

Optional: print shapes + split/label distributions

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print('CASAS windows:', df.shape); print(df.columns)"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/sphere_windows.parquet'); print('SPHERE windows:', df.shape); print(df['split'].value_counts()); print(df['label'].value_counts().head(10))"
```

---

## Phase 1 — Commit / Push (Code Only, No Data)

### 1.7) Confirm data is not staged

**Purpose:** ensure you are not committing `data/raw` or `data/processed`.

```powershell
git status
```

### 1.8) Stage changed source/config/docs files

**Purpose:** commit only pipeline code and config updates.

```powershell
git add README.md
git add config/base.yaml
git add neurosymbolic_iot/cli/preprocess.py
git add neurosymbolic_iot/data_processing/preprocess_casas.py
git add neurosymbolic_iot/data_processing/preprocess_sphere.py
git add neurosymbolic_iot/data_processing/splits.py
git add .gitignore
```

### 1.9) Commit

**Purpose:** snapshot a stable Phase 0–1 state.

```powershell
git commit -m "Phase 1: reproducible preprocessing for CASAS and SPHERE"
```

### 1.10) Push to GitHub

**Purpose:** publish the reproducible Phase 0–1 pipeline.

```powershell
git push
```

If upstream is not set:

```powershell
git push -u origin main
```

---

## Outputs Summary (What Phase 1 Produces)

* CASAS:
  * `data/processed/casas_windows.parquet` (windows + features + label)
  * `data/processed/casas_meta.json` (counts + timings + parameters)
* SPHERE:
  * `data/processed/sphere_windows.parquet` (windows + features + label + split)
  * `data/processed/sphere_meta.json` (counts + timings + parameters)

These files are local artifacts and should not be committed.

```

```
