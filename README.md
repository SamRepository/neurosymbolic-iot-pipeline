# Neuro-Symbolic IoT Pipeline (Reproducible Experiments)

This repository contains the experimental setup and evaluation code skeleton for:
**Building Smarter IoT Systems: A Neuro-Symbolic Approach for Data Federation and Real-Time Processing**

## Quickstart (Phase 0 + Phase 1)

1) Create a virtual environment and install deps:
```bash
pip install -r requirements.txt
```

2) Put datasets under:
- `data/raw/casas/` (CASAS event logs)
- `data/raw/sphere/` (SPHERE files)

3) Run preprocessing:
```bash
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset casas
python -m neurosymbolic_iot.cli.preprocess --config config/base.yaml --dataset sphere
```

Outputs are written to:
- `data/processed/casas_windows.parquet` + `data/processed/casas_meta.json`
- `data/processed/sphere_windows.parquet` + `data/processed/sphere_meta.json`

## Notes
- This is a skeleton focusing on reproducible data preparation (Phase 0 + Phase 1).
- Neural models, KG building, reasoning, and feedback loops are added in subsequent phases.
