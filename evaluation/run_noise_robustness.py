"""
Noise Robustness Experiment (Section 4.2)
=========================================
Compares GRU activity-recognition performance on CASAS Kyoto ADL subsets:
  - adl_error   : scripted tasks WITH injected sensor errors
  - adl_noerror : scripted tasks WITHOUT errors (clean baseline)
  - combined    : both subsets pooled (the default pipeline run)

Runs stratified 5-fold CV on each subset independently, then reports
accuracy, F1-macro, F1-weighted, and the performance delta.

Usage (PowerShell):
  $env:PYTHONPATH="."; python evaluation/run_noise_robustness.py --config config/base.yaml --k 5

Usage (Bash):
  PYTHONPATH=. python evaluation/run_noise_robustness.py --config config/base.yaml --k 5
"""
from __future__ import annotations

import argparse
import copy
import json
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

from neurosymbolic_iot.neural_perception.casas_sequence import (
    build_casas_sequences,
    build_casas_windows_from_raw,
)
from neurosymbolic_iot.neural_perception.cross_validation import run_kfold_cv
from neurosymbolic_iot.neural_perception.models import CasasGRUClassifier
from neurosymbolic_iot.neural_perception.utils import ensure_dir, save_json

log = logging.getLogger(__name__)


def _ensure_utc_tz(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Force a column to tz-aware UTC."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _build_windows_for_subset(
    cfg: Dict[str, Any],
    file_globs: List[str],
) -> pd.DataFrame:
    """Build CASAS windows using only the specified file_globs."""
    cfg_sub = copy.deepcopy(cfg)
    cfg_sub["datasets"]["casas"]["file_globs"] = file_globs

    np_cfg = cfg.get("neural_perception", {}).get("casas", {})
    casas_ds = cfg.get("datasets", {}).get("casas", {})

    dfw = build_casas_windows_from_raw(
        cfg_sub,
        window_minutes=int(np_cfg.get("window_minutes", casas_ds.get("window_minutes", 30))),
        stride_minutes=int(np_cfg.get("stride_minutes", casas_ds.get("stride_minutes", 5))),
        min_events=int(np_cfg.get("min_events", casas_ds.get("min_events_per_window", 1))),
    )
    for c in ("start_time", "end_time"):
        if c in dfw.columns:
            dfw = _ensure_utc_tz(dfw, c)
    return dfw


def run_casas_subset_cv(
    cfg: Dict[str, Any],
    dfw: pd.DataFrame,
    subset_name: str,
    k: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run k-fold CV on a CASAS window subset."""
    from neurosymbolic_iot.cli.train_neural import CasasTorchDataset, casas_collate

    np_cfg = cfg.get("neural_perception", {}).get("casas", {})
    max_seq_len = int(np_cfg.get("max_seq_len", 256))
    batch_size = int(np_cfg.get("batch_size", 32))
    emb_dim = int(np_cfg.get("emb_dim", np_cfg.get("embedding_dim", 64)))
    hidden = int(np_cfg.get("hidden_size", 128))
    num_layers = int(np_cfg.get("num_layers", 1))
    dropout = float(np_cfg.get("dropout", 0.1))

    labels = sorted(dfw["label"].astype(str).unique().tolist())
    id2label = labels

    # Build vocab from ALL windows in this subset (no leakage -- vocab is just token names)
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

    log.info(
        "%s: %d windows, %d classes, %d-fold CV",
        subset_name, len(dfw), len(labels), k,
    )

    return run_kfold_cv(
        build_dataset_fn=build_dataset_fn,
        build_model_fn=build_model_fn,
        df_all=dfw,
        label_col="label",
        id2label=id2label,
        cfg=cfg,
        out_dir=out_dir / f"{subset_name}_cv",
        k=k,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Noise Robustness: compare CASAS adl_error vs adl_noerror via k-fold CV.",
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--outdir", default="outputs/noise_robustness", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    # -- Build windows for each subset separately --
    log.info("=" * 60)
    log.info("Building windows: adl_error")
    log.info("=" * 60)
    dfw_error = _build_windows_for_subset(cfg, ["adl_error/*.csv"])

    log.info("=" * 60)
    log.info("Building windows: adl_noerror")
    log.info("=" * 60)
    dfw_noerror = _build_windows_for_subset(cfg, ["adl_noerror/*.csv"])

    log.info("=" * 60)
    log.info("Building windows: combined")
    log.info("=" * 60)
    dfw_combined = _build_windows_for_subset(
        cfg, ["adl_error/*.csv", "adl_noerror/*.csv"],
    )

    # -- Report dataset statistics --
    subsets = {
        "adl_error": dfw_error,
        "adl_noerror": dfw_noerror,
        "combined": dfw_combined,
    }
    for name, dfw in subsets.items():
        dist = dfw["label"].value_counts().to_dict()
        log.info("%s: %d windows, class distribution: %s", name, len(dfw), dist)

    # -- Run k-fold CV on each --
    results = {}
    for name, dfw in subsets.items():
        log.info("=" * 60)
        log.info("Running %d-fold CV: %s (%d windows)", args.k, name, len(dfw))
        log.info("=" * 60)
        results[name] = run_casas_subset_cv(
            cfg, dfw, name, args.k, seed, out_dir,
        )

    # -- Save combined results --
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "n_windows": len(subsets[name]),
            "class_distribution": subsets[name]["label"].value_counts().to_dict(),
            **r["aggregate"],
            "per_fold_accuracy": [f["val_accuracy"] for f in r["folds"]],
        }

    save_json(out_dir / "noise_robustness_results.json", summary)

    # -- Print comparison table --
    print("\n" + "=" * 78)
    print("  NOISE ROBUSTNESS EXPERIMENT -- CASAS GRU 5-Fold CV")
    print("=" * 78)
    header = f"  {'Subset':<14} {'N':>5} {'Accuracy':>18} {'F1-macro':>18} {'F1-weighted':>18}"
    print(header)
    print("-" * 78)

    for name in ["adl_noerror", "adl_error", "combined"]:
        s = summary[name]
        acc = f"{s['accuracy_mean']:.2%} +/- {s['accuracy_std']:.2%}"
        f1m = f"{s['f1_macro_mean']:.2%} +/- {s['f1_macro_std']:.2%}"
        f1w = f"{s['f1_weighted_mean']:.2%} +/- {s['f1_weighted_std']:.2%}"
        print(f"  {name:<14} {s['n_windows']:>5} {acc:>18} {f1m:>18} {f1w:>18}")

    # Delta row
    e = summary["adl_error"]
    n = summary["adl_noerror"]
    d_acc = e["accuracy_mean"] - n["accuracy_mean"]
    d_f1m = e["f1_macro_mean"] - n["f1_macro_mean"]
    d_f1w = e["f1_weighted_mean"] - n["f1_weighted_mean"]
    print("-" * 78)
    print(f"  {'Delta (E-N)':<14} {'':>5} {d_acc:>+18.2%} {d_f1m:>+18.2%} {d_f1w:>+18.2%}")
    print("=" * 78)

    print(f"\nResults saved to: {out_dir / 'noise_robustness_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
