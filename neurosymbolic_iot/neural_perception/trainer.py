from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from neurosymbolic_iot.neural_perception.utils import (
    compute_metrics,
    ensure_dir,
    save_confusion_csv,
    save_json,
)

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_f1_macro: float
    metrics: Dict


def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights with fixed size = num_classes.

    - Present classes get weight ~ 1/count.
    - Missing classes (count=0) get weight 0.0 (they will not contribute to loss).
    """
    y = y.astype(int)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float32)

    w = np.zeros(int(num_classes), dtype=np.float32)
    present = counts > 0

    if np.any(present):
        w[present] = 1.0 / counts[present]
        # Normalize so the average effective weight ~= 1 over present classes
        denom = float(np.sum(w[present] * counts[present]))
        scale = float(np.sum(counts[present])) / max(denom, 1e-8)
        w[present] *= scale

    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate_classifier(model, loader, device: str, id2label: List[str]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []

    for batch in loader:
        batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
        logits, y = model_forward(model, batch)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        ys.append(y.cpu().numpy())
        ps.append(pred)

    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=int)

    true_lbl = np.array([id2label[i] for i in y_true], dtype=object)
    pred_lbl = np.array([id2label[i] for i in y_pred], dtype=object)

    return compute_metrics(true_lbl, pred_lbl), y_true, y_pred


def model_forward(model, batch):
    """
    Standardize forward signatures:
      - CASAS: (token_ids, time_deltas, lengths, y)
      - SPHERE: (seq, y)
    """
    if len(batch) == 4:
        token_ids, time_deltas, lengths, y = batch
        logits = model(token_ids, time_deltas, lengths)
        return logits, y
    if len(batch) == 2:
        seq, y = batch
        logits = model(seq)
        return logits, y
    raise RuntimeError(f"Unexpected batch format (len={len(batch)}).")


def train_loop(
    *,
    model,
    train_loader,
    val_loader,
    test_loader,
    device: str,
    id2label: List[str],
    out_dir: Path,
    epochs: int,
    lr: float,
    seed: int,
) -> TrainResult:
    ensure_dir(out_dir)

    model = model.to(device)

    # Collect train labels to compute class weights robustly
    y_train = []
    for batch in train_loader:
        y = batch[-1]
        y_train.append(y.numpy())
    y_train = np.concatenate(y_train).astype(int)

    weights = _class_weights(y_train, num_classes=len(id2label)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    best_state = None
    best_epoch = -1
    best_val = -1.0

    history = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in train_loader:
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            logits, y = model_forward(model, batch)

            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * int(y.shape[0])
            n += int(y.shape[0])

        train_loss = total_loss / max(n, 1)

        val_metrics, yv, pv = (
            evaluate_classifier(model, val_loader, device, id2label) if val_loader else ({}, None, None)
        )
        val_f1 = float(val_metrics.get("f1_macro", 0.0)) if val_metrics else 0.0

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
        log.info("Epoch %d | train_loss=%.4f | val_f1_macro=%.4f", epoch, train_loss, val_f1)

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore best epoch weights
    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, yv, pv = (
        evaluate_classifier(model, val_loader, device, id2label) if val_loader else ({}, None, None)
    )
    test_metrics, yt, pt = (
        evaluate_classifier(model, test_loader, device, id2label) if test_loader else ({}, None, None)
    )

    metrics = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_val),
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }

    # Confusion matrices
    if yv is not None and pv is not None:
        save_confusion_csv(
            np.array([id2label[i] for i in yv], dtype=object),
            np.array([id2label[i] for i in pv], dtype=object),
            labels=id2label,
            out_csv=out_dir / "confusion_val.csv",
        )
    if yt is not None and pt is not None:
        save_confusion_csv(
            np.array([id2label[i] for i in yt], dtype=object),
            np.array([id2label[i] for i in pt], dtype=object),
            labels=id2label,
            out_csv=out_dir / "confusion_test.csv",
        )

    save_json(out_dir / "metrics.json", metrics)

    return TrainResult(best_epoch=best_epoch, best_val_f1_macro=best_val, metrics=metrics)
