from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

from neurosymbolic_iot.neural_perception.casas_sequence import (
    build_casas_sequences,
    build_casas_windows_from_raw,
)
from neurosymbolic_iot.neural_perception.models import CasasGRUClassifier, SphereLSTMClassifier
from neurosymbolic_iot.neural_perception.sphere_sequence import build_sphere_labeled_stream, build_sphere_windows
from neurosymbolic_iot.neural_perception.trainer import train_loop
from neurosymbolic_iot.neural_perception.utils import ensure_dir, pick_device, safe_split_df, save_json

log = logging.getLogger(__name__)


# -------------------------
# Utilities
# -------------------------
def _utc_tag() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _infer_group_cols(df: pd.DataFrame) -> List[str]:
    """
    Best-effort guess for grouping windows into independent timelines.
    If none found, we compute transition globally (warn).
    """
    candidates = [
        "source_file",
        "file",
        "filename",
        "series_id",
        "session_id",
        "participant",
        "subject",
        "home",
        "run_id",
    ]
    return [c for c in candidates if c in df.columns]


def _infer_sort_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer time columns when available
    if "start_time" in df.columns:
        return "start_time"
    if "window_start" in df.columns:
        return "window_start"
    # Fallback (not ideal)
    if "window_id" in df.columns:
        return "window_id"
    return None


def _ensure_utc_tz(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Force a column to be tz-aware UTC (datetime64[ns, UTC]) so comparisons against
    tz-aware event timestamps work (prevents: Cannot compare tz-naive and tz-aware).
    """
    if col not in df.columns:
        raise KeyError(f"Missing time column '{col}'. Available columns: {list(df.columns)}")

    out = df.copy()
    s = pd.to_datetime(out[col], utc=True, errors="coerce")  # always tz-aware UTC
    out[col] = s

    if out[col].isna().any():
        n_bad = int(out[col].isna().sum())
        raise ValueError(
            f"Time column '{col}' contains {n_bad} unparsable values after UTC coercion. "
            f"Check your preprocessing / raw data."
        )
    return out


def add_transition_labels(dfw: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures dfw has:
      - transition: int 0/1
      - label_transition: 'no_transition' or 'transition'

    If dfw already has 'transition', normalize it.
    Otherwise compute: transition = (label != previous_label) per timeline group.
    """
    df = dfw.copy()

    if "label" not in df.columns:
        raise ValueError("CASAS windows dataframe must contain a 'label' column to compute transitions.")

    df["label"] = df["label"].astype(str)

    if "transition" in df.columns:
        df["transition"] = pd.to_numeric(df["transition"], errors="coerce").fillna(0).astype(int)
        df["transition"] = df["transition"].clip(lower=0, upper=1)
    else:
        group_cols = _infer_group_cols(df)
        sort_col = _infer_sort_col(df)

        if sort_col is None:
            log.warning("CASAS: could not infer sort column; computing transition on current row order.")
        else:
            # If sorting by time, ensure tz-aware UTC to make ordering consistent
            if sort_col in ("start_time", "window_start", "end_time"):
                df = _ensure_utc_tz(df, sort_col)

        if not group_cols:
            log.warning(
                "CASAS: no grouping columns (e.g., file/session) found. "
                "Transition will be computed globally, which may mix independent timelines."
            )
            if sort_col:
                df = df.sort_values(by=[sort_col]).reset_index(drop=True)
            prev = df["label"].shift(1)
            df["transition"] = (df["label"] != prev).astype(int).fillna(0).astype(int)
            if len(df) > 0:
                df.loc[df.index[0], "transition"] = 0
        else:

            def _per_group(g: pd.DataFrame) -> pd.DataFrame:
                gg = g.copy()
                if sort_col:
                    gg = gg.sort_values(by=[sort_col])
                prev = gg["label"].shift(1)
                gg["transition"] = (gg["label"] != prev).astype(int).fillna(0).astype(int)
                if len(gg) > 0:
                    gg.iloc[0, gg.columns.get_loc("transition")] = 0
                return gg

            df = df.groupby(group_cols, group_keys=False).apply(_per_group).reset_index(drop=True)

    df["label_transition"] = df["transition"].map({0: "no_transition", 1: "transition"}).astype(str)
    return df


def ensure_all_labels_in_train(
    df_all: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Make sure every label in df_all[label_col] appears at least once in df_train[label_col].
    If missing, move 1 sample from VAL (preferred) else TEST into TRAIN.
    """
    all_labels = set(df_all[label_col].astype(str).unique().tolist())
    train_labels = set(df_train[label_col].astype(str).unique().tolist())
    missing = sorted(list(all_labels - train_labels))

    if not missing:
        return df_train, df_val, df_test

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    for lab in missing:
        moved = False
        mask_val = df_val[label_col].astype(str) == lab
        if mask_val.any():
            row = df_val[mask_val].head(1)
            df_val = df_val.drop(index=row.index)
            df_train = pd.concat([df_train, row], ignore_index=True)
            moved = True
        else:
            mask_test = df_test[label_col].astype(str) == lab
            if mask_test.any():
                row = df_test[mask_test].head(1)
                df_test = df_test.drop(index=row.index)
                df_train = pd.concat([df_train, row], ignore_index=True)
                moved = True

        if moved:
            log.warning("CASAS: moved 1 sample of label '%s' into TRAIN to avoid missing-class issues.", lab)
        else:
            log.warning("CASAS: could not move label '%s' into TRAIN (not found in VAL/TEST).", lab)

    return df_train, df_val, df_test


# -------------------------
# CASAS Torch dataset + collate
# -------------------------
class CasasTorchDataset(Dataset):
    def __init__(self, seqs, label2id: Dict[str, int], max_seq_len: int):
        self.seqs = seqs
        self.label2id = label2id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        s = self.seqs[idx]
        token_ids = np.array(s.token_ids, dtype=np.int64)
        time_deltas = np.array(s.time_deltas, dtype=np.float32)
        length = int(len(token_ids))
        y = int(self.label2id[str(s.label)])
        return token_ids, time_deltas, length, y


def casas_collate(batch):
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    token_pad = torch.zeros((len(batch), max_len), dtype=torch.long)
    td_pad = torch.zeros((len(batch), max_len), dtype=torch.float32)
    y = torch.tensor([b[3] for b in batch], dtype=torch.long)

    for i, (tok, td, L, _) in enumerate(batch):
        tok_t = torch.tensor(tok[:max_len], dtype=torch.long)
        td_t = torch.tensor(td[:max_len], dtype=torch.float32)
        token_pad[i, :L] = tok_t

        # normalize time by window duration for stability
        if L > 0:
            denom = max(float(td_t[-1].item()), 1.0)
            td_pad[i, :L] = td_t / denom

    return token_pad, td_pad, lengths, y


# -------------------------
# SPHERE Torch dataset
# -------------------------
class SphereTorchDataset(Dataset):
    def __init__(self, windows, label2id: Dict[str, int]):
        self.windows = windows
        self.label2id = label2id

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x = torch.tensor(w.seq, dtype=torch.float32)
        y = torch.tensor(self.label2id[str(w.label)], dtype=torch.long)
        return x, y


# -------------------------
# Main
# -------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2B: Train neural perception models (GRU/LSTM).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["casas", "sphere"])
    parser.add_argument(
        "--task",
        default="activity",
        choices=["activity", "transition"],
        help="CASAS only: activity (multi-class) or transition (binary). Default: activity.",
    )
    parser.add_argument("--tag", default=None, help="Run tag. Default = UTC timestamp.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    np_cfg = cfg.get("neural_perception", {})
    out_base = Path(np_cfg.get("output_dir", "outputs/neural_perception"))
    tag = args.tag or _utc_tag()

    # Separate outputs by task for CASAS (avoid mixing activity-vs-transition artifacts)
    if args.dataset == "casas":
        out_dir = out_base / tag / "casas" / args.task
    else:
        out_dir = out_base / tag / "sphere"

    ensure_dir(out_dir)

    device = pick_device(str(np_cfg.get("device", "auto")))
    log.info("Using device: %s", device)
    log.info("Run output dir: %s", out_dir.as_posix())

    # -------------------------
    # CASAS (GRU)
    # -------------------------
    if args.dataset == "casas":
        dcfg = np_cfg.get("casas", {})

        # We require datasets.casas because build_casas_windows_from_raw depends on it
        casas_ds = cfg.get("datasets", {}).get("casas", {})
        if not casas_ds:
            raise KeyError(
                "Config is missing datasets.casas. Use config/base.yaml or add:\n"
                "datasets:\n"
                "  casas:\n"
                "    raw_dir: ...\n"
                "    format: kyoto_adl_errors\n"
                "    file_globs: [...]\n"
            )

        window_minutes = int(dcfg.get("window_minutes", casas_ds.get("window_minutes", 30)))
        stride_minutes = int(dcfg.get("stride_minutes", casas_ds.get("stride_minutes", 5)))
        min_events = int(dcfg.get("min_events", casas_ds.get("min_events_per_window", 1)))

        max_seq_len = int(dcfg.get("max_seq_len", 256))
        emb_dim = int(dcfg.get("embedding_dim", 64))
        hidden = int(dcfg.get("hidden_size", 128))
        num_layers = int(dcfg.get("num_layers", 1))
        dropout = float(dcfg.get("dropout", 0.1))
        batch_size = int(dcfg.get("batch_size", 32))
        epochs = int(dcfg.get("epochs", 30))
        lr = float(dcfg.get("lr", 1e-3))

        train_ratio = float(dcfg.get("train_ratio", 0.7))
        val_ratio = float(dcfg.get("val_ratio", 0.15))
        test_ratio = float(dcfg.get("test_ratio", 0.15))

        # --- CASAS raw_dir verification (Kyoto ADL errors) ---
        casas_ds = cfg.get("datasets", {}).get("casas", {})
        raw_dir = Path(str(casas_ds.get("raw_dir", ""))).expanduser()
        raw_dir_abs = raw_dir.resolve() if raw_dir.exists() else raw_dir

        fmt = str(casas_ds.get("format", ""))
        globs = casas_ds.get("file_globs", [])

        log.info("CASAS config: format=%s", fmt)
        log.info("CASAS config: raw_dir=%s", raw_dir_abs.as_posix())
        log.info("CASAS config: file_globs=%s", globs)

        # Hard guarantee for kyoto_adl_errors layout
        if fmt == "kyoto_adl_errors":
            err_dir = raw_dir / "adl_error"
            noerr_dir = raw_dir / "adl_noerror"
            if not err_dir.exists() or not noerr_dir.exists():
                raise FileNotFoundError(
                    "CASAS kyoto_adl_errors expects:\n"
                    f"  {err_dir}\n"
                    f"  {noerr_dir}\n"
                    "But one (or both) is missing. Fix datasets.casas.raw_dir in your config."
                )

        # Build windows from raw
        dfw = build_casas_windows_from_raw(
            cfg,
            window_minutes=window_minutes,
            stride_minutes=stride_minutes,
            min_events=min_events,
        )

        # Critical: ensure window times are tz-aware UTC so build_casas_sequences() comparisons work
        for c in ("start_time", "end_time"):
            if c in dfw.columns:
                dfw = _ensure_utc_tz(dfw, c)

        # Transition task: compute transition labels then switch dfw['label'] to binary class
        if args.task == "transition":
            dfw = add_transition_labels(dfw)
            dfw["activity_label"] = dfw["label"].astype(str)  # preserve original activity label
            dfw["label"] = dfw["label_transition"].astype(str)

            tr_rate = float(dfw["transition"].mean()) if "transition" in dfw.columns else float("nan")
            log.info("CASAS transition task: transition_rate=%.4f (n=%d)", tr_rate, len(dfw))

        # Split (stratify by the current dfw['label'])
        splits = safe_split_df(
            dfw,
            train=train_ratio,
            val=val_ratio,
            test=test_ratio,
            seed=seed,
            stratify_col="label",
        )

        df_train = splits.train.copy()
        df_val = splits.val.copy()
        df_test = splits.test.copy()

        # Ensure all labels appear in TRAIN
        df_train, df_val, df_test = ensure_all_labels_in_train(dfw, df_train, df_val, df_test, label_col="label")

        # Ensure split dfs still have tz-aware UTC times (some operations can cast)
        for split_df_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            for c in ("start_time", "end_time"):
                if c in split_df.columns:
                    coerced = _ensure_utc_tz(split_df, c)
                    if split_df_name == "train":
                        df_train = coerced
                    elif split_df_name == "val":
                        df_val = coerced
                    else:
                        df_test = coerced

        # Build vocab from TRAIN only to avoid leakage
        train_seqs, vocab = build_casas_sequences(cfg, df_train, max_seq_len=max_seq_len, vocab=None)
        val_seqs, _ = build_casas_sequences(cfg, df_val, max_seq_len=max_seq_len, vocab=vocab)
        test_seqs, _ = build_casas_sequences(cfg, df_test, max_seq_len=max_seq_len, vocab=vocab)

        # Label space
        if args.task == "transition":
            # Force stable label ordering for binary task
            labels = ["no_transition", "transition"]
            # But keep only those that actually exist (in case of weird edge-case splits)
            present = sorted({str(s.label) for s in train_seqs})
            labels = [x for x in labels if x in present] + [x for x in present if x not in labels]
        else:
            labels = sorted({str(s.label) for s in train_seqs})

        label2id = {lab: i for i, lab in enumerate(labels)}
        id2label = labels

        save_json(out_dir / "label_map.json", {"label2id": label2id, "id2label": id2label})
        save_json(out_dir / "vocab.json", vocab)
        save_json(
            out_dir / "task.json",
            {
                "dataset": "casas",
                "task": args.task,
                "window_minutes": window_minutes,
                "stride_minutes": stride_minutes,
                "min_events": min_events,
                "max_seq_len": max_seq_len,
                "seed": seed,
                "n_train": len(df_train),
                "n_val": len(df_val),
                "n_test": len(df_test),
            },
        )

        ds_train = CasasTorchDataset(train_seqs, label2id, max_seq_len)
        ds_val = CasasTorchDataset(val_seqs, label2id, max_seq_len)
        ds_test = CasasTorchDataset(test_seqs, label2id, max_seq_len)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=casas_collate)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)

        model = CasasGRUClassifier(
            vocab_size=len(vocab),
            num_classes=len(labels),  # 2 for transition, N for activity
            emb_dim=emb_dim,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
        )

        train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            id2label=id2label,
            out_dir=out_dir,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )

        # Save model
        torch.save(model.state_dict(), out_dir / "model.pth")
        log.info("Saved CASAS model to %s", (out_dir / "model.pth").as_posix())
        log.info("Metrics written to: %s", (out_dir / "metrics.json").as_posix())
        return 0

    # -------------------------
    # SPHERE (LSTM)
    # -------------------------
    if args.task != "activity":
        log.warning("SPHERE ignores --task=%s (only activity classification is supported).", args.task)

    dcfg = np_cfg.get("sphere", {})

    sphere_ds = cfg.get("datasets", {}).get("sphere", {})
    if not sphere_ds:
        raise KeyError(
            "Config is missing datasets.sphere. Use config/base.yaml or add a datasets.sphere section."
        )

    window_seconds = int(dcfg.get("window_seconds", sphere_ds.get("window_seconds", 30)))
    stride_seconds = int(dcfg.get("stride_seconds", sphere_ds.get("stride_seconds", 15)))
    min_rows = int(dcfg.get("min_rows", sphere_ds.get("min_rows_per_window", 10)))
    seq_len = int(dcfg.get("seq_len", 128))

    hidden = int(dcfg.get("hidden_size", 128))
    num_layers = int(dcfg.get("num_layers", 1))
    dropout = float(dcfg.get("dropout", 0.2))
    batch_size = int(dcfg.get("batch_size", 64))
    epochs = int(dcfg.get("epochs", 40))
    lr = float(dcfg.get("lr", 1e-3))

    train_ratio = float(dcfg.get("train_ratio", 0.7))
    val_ratio = float(dcfg.get("val_ratio", 0.15))
    test_ratio = float(dcfg.get("test_ratio", 0.15))

    labeled = build_sphere_labeled_stream(cfg)
    windows = build_sphere_windows(
        labeled,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_rows=min_rows,
        seq_len=seq_len,
    )

    dfw = pd.DataFrame([{"idx": i, "label": str(w.label)} for i, w in enumerate(windows)])

    splits = safe_split_df(
        dfw,
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
        seed=seed,
        stratify_col="label",
    )

    df_train = splits.train.copy()
    df_val = splits.val.copy()
    df_test = splits.test.copy()

    # Ensure every label appears in TRAIN
    all_labels = set(dfw["label"].astype(str).unique().tolist())
    train_labels = set(df_train["label"].astype(str).unique().tolist())
    missing = sorted(list(all_labels - train_labels))

    for lab in missing:
        moved = False
        if (df_val["label"].astype(str) == lab).any():
            row = df_val[df_val["label"].astype(str) == lab].head(1)
            df_val = df_val.drop(index=row.index)
            df_train = pd.concat([df_train, row], ignore_index=True)
            moved = True
        elif (df_test["label"].astype(str) == lab).any():
            row = df_test[df_test["label"].astype(str) == lab].head(1)
            df_test = df_test.drop(index=row.index)
            df_train = pd.concat([df_train, row], ignore_index=True)
            moved = True

        if moved:
            log.warning("SPHERE: moved 1 sample of label '%s' into TRAIN to avoid missing-class issues.", lab)

    def pick(split_df: pd.DataFrame):
        return [windows[int(i)] for i in split_df["idx"].tolist()]

    train_w = pick(df_train)
    val_w = pick(df_val)
    test_w = pick(df_test)

    labels = sorted({str(w.label) for w in windows})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = labels

    save_json(out_dir / "label_map.json", {"label2id": label2id, "id2label": id2label})
    save_json(
        out_dir / "task.json",
        {
            "dataset": "sphere",
            "task": "activity",
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "min_rows": min_rows,
            "seq_len": seq_len,
            "seed": seed,
            "n_train": len(train_w),
            "n_val": len(val_w),
            "n_test": len(test_w),
        },
    )

    ds_train = SphereTorchDataset(train_w, label2id)
    ds_val = SphereTorchDataset(val_w, label2id)
    ds_test = SphereTorchDataset(test_w, label2id)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    in_dim = int(train_w[0].seq.shape[1])
    model = SphereLSTMClassifier(
        in_dim=in_dim,
        num_classes=len(labels),
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        id2label=id2label,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )

    torch.save(model.state_dict(), out_dir / "model.pth")
    log.info("Saved SPHERE model to %s", (out_dir / "model.pth").as_posix())
    log.info("Metrics written to: %s", (out_dir / "metrics.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
