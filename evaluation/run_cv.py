"""
5-fold cross-validation for CASAS (GRU) and SPHERE (LSTM) neural models.

Usage (PowerShell):
  $env:PYTHONPATH="."; python evaluation/run_cv.py --config config/base.yaml --datasets casas,sphere --k 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

from neurosymbolic_iot.neural_perception.cross_validation import run_kfold_cv
from neurosymbolic_iot.neural_perception.models import CasasGRUClassifier, SphereLSTMClassifier
from neurosymbolic_iot.neural_perception.utils import pick_device

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CASAS: wire up GRU data pipeline for CV
# ---------------------------------------------------------------------------

def run_casas_cv(cfg: Dict[str, Any], k: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    from neurosymbolic_iot.neural_perception.casas_sequence import (
        build_casas_sequences,
        build_casas_windows_from_raw,
    )
    from neurosymbolic_iot.cli.train_neural import (
        CasasTorchDataset,
        casas_collate,
        _ensure_utc_tz,
    )

    np_cfg = cfg.get("neural_perception", {}).get("casas", {})
    casas_ds = cfg.get("datasets", {}).get("casas", {})

    window_minutes = int(np_cfg.get("window_minutes", casas_ds.get("window_minutes", 30)))
    stride_minutes = int(np_cfg.get("stride_minutes", casas_ds.get("stride_minutes", 5)))
    min_events = int(np_cfg.get("min_events", casas_ds.get("min_events_per_window", 1)))
    max_seq_len = int(np_cfg.get("max_seq_len", 256))
    batch_size = int(np_cfg.get("batch_size", 32))
    emb_dim = int(np_cfg.get("emb_dim", np_cfg.get("embedding_dim", 64)))
    hidden = int(np_cfg.get("hidden_size", 128))
    num_layers = int(np_cfg.get("num_layers", 1))
    dropout = float(np_cfg.get("dropout", 0.1))

    # Build all windows
    dfw = build_casas_windows_from_raw(
        cfg,
        window_minutes=window_minutes,
        stride_minutes=stride_minutes,
        min_events=min_events,
    )
    for c in ("start_time", "end_time"):
        if c in dfw.columns:
            dfw = _ensure_utc_tz(dfw, c)

    labels = sorted(dfw["label"].astype(str).unique().tolist())
    id2label = labels

    # Build vocab from ALL data (needed by all folds; no leakage since vocab is just sensor tokens)
    _, full_vocab = build_casas_sequences(cfg, dfw, max_seq_len=max_seq_len, vocab=None)

    def build_dataset_fn(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        label2id: Dict[str, int],
    ) -> Tuple[DataLoader, DataLoader]:
        for c in ("start_time", "end_time"):
            if c in df_train.columns:
                df_train = _ensure_utc_tz(df_train, c)
            if c in df_val.columns:
                df_val = _ensure_utc_tz(df_val, c)

        train_seqs, _ = build_casas_sequences(cfg, df_train, max_seq_len=max_seq_len, vocab=full_vocab)
        val_seqs, _ = build_casas_sequences(cfg, df_val, max_seq_len=max_seq_len, vocab=full_vocab)

        ds_train = CasasTorchDataset(train_seqs, label2id, max_seq_len)
        ds_val = CasasTorchDataset(val_seqs, label2id, max_seq_len)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=casas_collate)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)
        return train_loader, val_loader

    def build_model_fn(num_classes: int) -> torch.nn.Module:
        return CasasGRUClassifier(
            vocab_size=len(full_vocab),
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
        )

    log.info("CASAS CV: %d windows, %d classes, %d-fold", len(dfw), len(labels), k)

    return run_kfold_cv(
        build_dataset_fn=build_dataset_fn,
        build_model_fn=build_model_fn,
        df_all=dfw,
        label_col="label",
        id2label=id2label,
        cfg=cfg,
        out_dir=out_dir / "casas_cv",
        k=k,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# SPHERE: wire up LSTM data pipeline for CV
# ---------------------------------------------------------------------------

def run_sphere_cv(cfg: Dict[str, Any], k: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    from neurosymbolic_iot.neural_perception.sphere_sequence import (
        build_sphere_labeled_stream,
        build_sphere_windows,
    )
    from neurosymbolic_iot.cli.train_neural import SphereTorchDataset

    np_cfg = cfg.get("neural_perception", {}).get("sphere", {})
    sphere_ds = cfg.get("datasets", {}).get("sphere", {})

    window_seconds = int(np_cfg.get("window_seconds", sphere_ds.get("window_seconds", 30)))
    stride_seconds = int(np_cfg.get("stride_seconds", sphere_ds.get("stride_seconds", 15)))
    min_rows = int(np_cfg.get("min_rows", sphere_ds.get("min_rows_per_window", 10)))
    seq_len = int(np_cfg.get("seq_len", 128))
    batch_size = int(np_cfg.get("batch_size", 64))
    hidden = int(np_cfg.get("hidden_size", 128))
    num_layers = int(np_cfg.get("num_layers", 1))
    dropout = float(np_cfg.get("dropout", 0.2))

    # Build all windows
    labeled = build_sphere_labeled_stream(cfg)
    windows = build_sphere_windows(
        labeled,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_rows=min_rows,
        seq_len=seq_len,
    )

    labels = sorted({str(w.label) for w in windows})
    id2label = labels
    in_dim = int(windows[0].seq.shape[1])

    # Create a DataFrame with index into windows list for fold splitting
    dfw = pd.DataFrame([{"idx": i, "label": str(w.label)} for i, w in enumerate(windows)])

    def build_dataset_fn(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        label2id: Dict[str, int],
    ) -> Tuple[DataLoader, DataLoader]:
        train_w = [windows[int(i)] for i in df_train["idx"].tolist()]
        val_w = [windows[int(i)] for i in df_val["idx"].tolist()]

        ds_train = SphereTorchDataset(train_w, label2id)
        ds_val = SphereTorchDataset(val_w, label2id)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def build_model_fn(num_classes: int) -> torch.nn.Module:
        return SphereLSTMClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
        )

    log.info("SPHERE CV: %d windows, %d classes, %d-fold", len(windows), len(labels), k)

    return run_kfold_cv(
        build_dataset_fn=build_dataset_fn,
        build_model_fn=build_model_fn,
        df_all=dfw,
        label_col="label",
        id2label=id2label,
        cfg=cfg,
        out_dir=out_dir / "sphere_cv",
        k=k,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation for neural models.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--datasets", default="casas,sphere", help="Comma-separated: casas,sphere")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--outdir", default="outputs/cross_validation", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    out_dir = Path(args.outdir)
    requested = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    results = {}

    if "casas" in requested:
        log.info("=" * 60)
        log.info("Starting CASAS %d-fold CV", args.k)
        log.info("=" * 60)
        results["casas"] = run_casas_cv(cfg, args.k, seed, out_dir)

    if "sphere" in requested:
        log.info("=" * 60)
        log.info("Starting SPHERE %d-fold CV", args.k)
        log.info("=" * 60)
        results["sphere"] = run_sphere_cv(cfg, args.k, seed, out_dir)

    # Print summary
    print("\n" + "=" * 70)
    print(f"  {args.k}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)
    for ds, r in results.items():
        agg = r["aggregate"]
        print(f"\n  {ds.upper()}:")
        print(f"    Accuracy:    {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
        print(f"    F1-macro:    {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
        print(f"    F1-weighted: {agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}")
        print(f"    Per-fold accuracy: {[round(f['val_accuracy'], 4) for f in r['folds']]}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
