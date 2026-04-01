from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neurosymbolic_iot.neural_perception.models import (
    CasasGRUClassifier,
    SphereLSTMClassifier,
)
from neurosymbolic_iot.neural_perception.trainer import model_forward
from neurosymbolic_iot.neural_perception.utils import ensure_dir, save_json

log = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """One per sample: predicted label, confidence, full probability vector."""

    sample_idx: int
    predicted_label: str
    confidence: float
    probabilities: List[float]
    ground_truth_label: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Batch of predictions from a single inference run."""

    predictions: List[PredictionRecord]
    id2label: List[str]
    model_tag: str
    dataset: str


def load_trained_model(
    model_dir: Path,
    device: str,
    dataset: str,
    cfg: Dict[str, Any],
) -> Tuple[torch.nn.Module, List[str], Dict[str, int], Dict[str, Any]]:
    """Load a trained model and its associated artifacts from *model_dir*."""
    model_dir = Path(model_dir)

    with open(model_dir / "label_map.json", encoding="utf-8") as f:
        lm = json.load(f)
    label2id: Dict[str, int] = lm["label2id"]
    id2label: List[str] = lm["id2label"]
    num_classes = len(id2label)

    with open(model_dir / "task.json", encoding="utf-8") as f:
        task_info: Dict[str, Any] = json.load(f)

    if dataset == "casas":
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab: Dict[str, int] = json.load(f)
        np_cfg = cfg.get("neural_perception", {}).get("casas", {})
        model = CasasGRUClassifier(
            vocab_size=len(vocab),
            num_classes=num_classes,
            emb_dim=int(np_cfg.get("emb_dim", 64)),
            hidden=int(np_cfg.get("hidden", 128)),
            num_layers=int(np_cfg.get("num_layers", 1)),
        )
    elif dataset == "sphere":
        np_cfg = cfg.get("neural_perception", {}).get("sphere", {})
        in_dim = int(task_info.get("in_dim", np_cfg.get("in_dim", 3)))
        model = SphereLSTMClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            hidden=int(np_cfg.get("hidden", 128)),
            num_layers=int(np_cfg.get("num_layers", 1)),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    state = torch.load(model_dir / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info("Loaded model from %s (%d classes, device=%s)", model_dir, num_classes, device)
    return model, id2label, label2id, task_info


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    id2label: List[str],
    *,
    window_metadata: Optional[List[Dict[str, Any]]] = None,
    model_tag: str = "unknown",
    dataset: str = "unknown",
) -> InferenceResult:
    """Run inference, returning per-sample predictions with softmax confidence."""
    model.eval()
    records: List[PredictionRecord] = []
    global_idx = 0

    for batch in loader:
        batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
        logits, y = model_forward(model, batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        y_np = y.cpu().numpy()

        for i in range(len(preds)):
            meta = window_metadata[global_idx] if window_metadata else {}
            records.append(
                PredictionRecord(
                    sample_idx=global_idx,
                    predicted_label=id2label[int(preds[i])],
                    confidence=float(probs[i, int(preds[i])]),
                    probabilities=probs[i].tolist(),
                    ground_truth_label=id2label[int(y_np[i])] if int(y_np[i]) < len(id2label) else None,
                    window_start=meta.get("window_start"),
                    window_end=meta.get("window_end"),
                    metadata=meta,
                )
            )
            global_idx += 1

    log.info("Inference complete: %d samples, model=%s", len(records), model_tag)
    return InferenceResult(predictions=records, id2label=id2label, model_tag=model_tag, dataset=dataset)


def save_predictions(result: InferenceResult, out_path: Path) -> None:
    """Serialize predictions to JSON."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    data = {
        "model_tag": result.model_tag,
        "dataset": result.dataset,
        "num_predictions": len(result.predictions),
        "id2label": result.id2label,
        "predictions": [asdict(p) for p in result.predictions],
    }
    save_json(out_path, data)
    log.info("Saved %d predictions to %s", len(result.predictions), out_path)
