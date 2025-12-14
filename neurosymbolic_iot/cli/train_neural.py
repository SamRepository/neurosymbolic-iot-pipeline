from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List

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


def _utc_tag() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


# -------------------------
# CASAS Dataset + collate
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
        y = int(self.label2id[s.label])
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
# SPHERE Dataset
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
        # robust (label2id now contains ALL labels)
        y = torch.tensor(self.label2id[str(w.label)], dtype=torch.long)
        return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2B: Train neural perception models (GRU/LSTM).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["casas", "sphere"])
    parser.add_argument("--tag", default=None, help="Run tag. Default = UTC timestamp.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    np_cfg = cfg.get("neural_perception", {})
    out_base = Path(np_cfg.get("output_dir", "outputs/neural_perception"))
    tag = args.tag or _utc_tag()
    out_dir = out_base / tag / args.dataset
    ensure_dir(out_dir)

    device = pick_device(str(np_cfg.get("device", "auto")))
    log.info("Using device: %s", device)

    # -------------------------
    # CASAS (GRU)
    # -------------------------
    if args.dataset == "casas":
        dcfg = np_cfg.get("casas", {})

        window_minutes = int(dcfg.get("window_minutes", cfg["datasets"]["casas"].get("window_minutes", 30)))
        stride_minutes = int(dcfg.get("stride_minutes", cfg["datasets"]["casas"].get("stride_minutes", 5)))
        min_events = int(dcfg.get("min_events", cfg["datasets"]["casas"].get("min_events_per_window", 5)))

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

        dfw = build_casas_windows_from_raw(
            cfg,
            window_minutes=window_minutes,
            stride_minutes=stride_minutes,
            min_events=min_events,
        )

        splits = safe_split_df(
            dfw,
            train=train_ratio,
            val=val_ratio,
            test=test_ratio,
            seed=seed,
            stratify_col="label",
        )

        # build vocab from TRAIN only to avoid leakage
        train_seqs, vocab = build_casas_sequences(cfg, splits.train, max_seq_len=max_seq_len, vocab=None)
        val_seqs, _ = build_casas_sequences(cfg, splits.val, max_seq_len=max_seq_len, vocab=vocab)
        test_seqs, _ = build_casas_sequences(cfg, splits.test, max_seq_len=max_seq_len, vocab=vocab)

        labels = sorted({s.label for s in train_seqs})
        label2id = {lab: i for i, lab in enumerate(labels)}
        id2label = labels

        save_json(out_dir / "label_map.json", {"label2id": label2id, "id2label": id2label})
        save_json(out_dir / "vocab.json", vocab)

        ds_train = CasasTorchDataset(train_seqs, label2id, max_seq_len)
        ds_val = CasasTorchDataset(val_seqs, label2id, max_seq_len)
        ds_test = CasasTorchDataset(test_seqs, label2id, max_seq_len)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=casas_collate)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)

        model = CasasGRUClassifier(
            vocab_size=len(vocab),
            num_classes=len(labels),
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

        torch.save(model.state_dict(), out_dir / "model.pth")
        log.info("Saved CASAS model to %s", (out_dir / "model.pth").as_posix())
        log.info("Metrics written to: %s", (out_dir / "metrics.json").as_posix())
        return 0

    # -------------------------
    # SPHERE (LSTM)
    # -------------------------
    dcfg = np_cfg.get("sphere", {})

    window_seconds = int(dcfg.get("window_seconds", cfg["datasets"]["sphere"].get("window_seconds", 30)))
    stride_seconds = int(dcfg.get("stride_seconds", cfg["datasets"]["sphere"].get("stride_seconds", 15)))
    min_rows = int(dcfg.get("min_rows", cfg["datasets"]["sphere"].get("min_rows_per_window", 10)))
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

    # Make split DataFrames mutable copies
    df_train = splits.train.copy()
    df_val = splits.val.copy()
    df_test = splits.test.copy()

    # Ensure every label appears at least once in TRAIN
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

    # Build label space from ALL windows (train+val+test) to prevent KeyError
    labels = sorted({str(w.label) for w in windows})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = labels

    save_json(out_dir / "label_map.json", {"label2id": label2id, "id2label": id2label})

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
