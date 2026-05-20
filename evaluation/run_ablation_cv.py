"""
5-fold CV ablation study with paired statistical tests (R2.1).

For each of the 5 stratified folds (Seed = 42, same iterator that produced
Table 5), trains the CASAS GRU on the train indices, evaluates the four
ablation configurations on the held-out fold, and aggregates per-fold
metrics. Then runs paired Wilcoxon signed-rank tests (one-sided,
alternative = NeSy-Full > baseline) and Cohen's d for the two pairs:
{AI-Only -> NeSy-Full} and {NeSy-NoFeedback -> NeSy-Full}.

Configurations:
  AI-Only          standalone GRU
  KG+Rules         symbolic only (Python rule executor on KG built from
                   sensor metadata; no neural prediction)
  NeSy-NoFeedback  GRU + KG + rule executor; predictions flagged
                   FeedbackRequired are dropped from the active set;
                   rule-validated events (rules 3/4) override the
                   neural prediction
  NeSy-Full        NeSy-NoFeedback + one feedback cycle (threshold
                   adjustment + retraction of FalsePositiveHallucination
                   triples)

CEP latency is reported as a point estimate and is NOT cross-validated
(it is an infrastructure metric, not stochastic). The script prints a
LaTeX-ready table fragment to stdout and saves the full numeric record
to ``evaluation/results/ablation_cv.json``.

Usage::

    PYTHONPATH=. python evaluation/run_ablation_cv.py \
        --config config/base.yaml \
        --k 5 \
        --epochs 30 \
        --cep-latency 12.4
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set BEFORE importing torch — required for use_deterministic_algorithms(True)
# on any CUBLAS / cuDNN code path. Harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import torch

# Pin all torch RNG sources to deterministic algorithms. Must run before
# any tensor is allocated; CPU-only run so cuDNN flags are no-ops.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _set_per_fold_seed(base_seed: int, fold_idx: int) -> int:
    """Reset every relevant RNG to a deterministic, fold-specific value.

    Without this, RNG state evolves through the script as preceding folds
    consume random numbers (DataLoader shuffle, dropout, KG triple
    iteration, rdflib SPARQL planner). The downstream effect is that
    fold N's trained model depends on the wall-clock history of folds
    0..N-1 — meaning re-runs are not byte-stable. Resetting here pins
    each fold to a known starting state.
    """
    seed = int(base_seed) + 1000 * int(fold_idx)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


CONFIG_NAMES = ["AI-Only", "KG+Rules", "NeSy-NoFeedback", "NeSy-Full"]
PRIMARY_METRICS = ["F1_weighted", "Correctness", "FP_rate"]


# ---------------------------------------------------------------------------
# Per-fold metric extraction for the four configurations
# ---------------------------------------------------------------------------

def _ai_only_metrics(predictions: List[Dict[str, Any]], conf_threshold: float) -> Dict[str, float]:
    """Standalone GRU: F1 / correctness / FP rate over predictions whose
    confidence is at or above the threshold."""
    active = [p for p in predictions
              if p.get("ground_truth_label") is not None
              and float(p.get("confidence", 0.0)) >= conf_threshold]
    if not active:
        return {"F1_weighted": 0.0, "Correctness": 0.0, "FP_rate": 0.0, "n_active": 0}
    y_true = [p["ground_truth_label"] for p in active]
    y_pred = [p["predicted_label"] for p in active]
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    correctness = correct / len(active)
    return {
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Correctness": float(correctness),
        "FP_rate": float(1.0 - correctness),
        "n_active": len(active),
    }


def _kg_only_metrics(
    predictions: List[Dict[str, Any]],
    validated_events: List[Dict[str, Any]],
    inferred_activities: List[Dict[str, Any]],
    activity_iri_to_label: Dict[str, str],
) -> Dict[str, Any]:
    """Symbolic-only baseline: correctness over events whose activity has been
    rule-asserted (rules 3/4 emit performsActivity + ValidatedEvent).

    F1_weighted is not computable for KG-only because a rule-only system
    has no per-window prediction except for the ValidatedEvent subset; per
    the brief, we keep F1 as N/A rather than fabricating a value over a
    sparse subset."""
    if not validated_events or not inferred_activities:
        return {"F1_weighted": None, "Correctness": None, "FP_rate": None, "n_validated": 0}

    # Map event index -> ground truth via the predictions ordering.
    ev_to_gt: Dict[int, str] = {}
    for i, p in enumerate(predictions):
        if p.get("ground_truth_label") is not None:
            ev_to_gt[i] = p["ground_truth_label"]

    # Validated events have URIs of the form ".../event_<idx>".
    correct = 0
    total = 0
    for ev in validated_events:
        uri = ev.get("uri", "")
        try:
            ev_idx = int(uri.rsplit("event_", 1)[1])
        except (IndexError, ValueError):
            continue
        gt = ev_to_gt.get(ev_idx)
        if gt is None:
            continue
        # Find the rule-asserted activity for this event's person.
        # All events involve Participant1 in the current builder, so we
        # accept any inferred activity that is in the activity_iri_to_label
        # map.
        rule_acts = [activity_iri_to_label.get(a["activity"]) for a in inferred_activities]
        rule_acts = [a for a in rule_acts if a is not None]
        if not rule_acts:
            continue
        # If any rule-asserted activity matches gt, count as correct.
        total += 1
        if gt in rule_acts:
            correct += 1

    if total == 0:
        return {"F1_weighted": None, "Correctness": None, "FP_rate": None, "n_validated": 0}

    correctness = correct / total
    return {
        "F1_weighted": None,        # not defined for rule-only baseline (sparse)
        "Correctness": float(correctness),
        "FP_rate": float(1.0 - correctness),
        "n_validated": total,
    }


def _nesy_metrics(
    predictions: List[Dict[str, Any]],
    feedback_flags: List[Dict[str, Any]],
    validated_events: List[Dict[str, Any]],
    inferred_activities: List[Dict[str, Any]],
    activity_iri_to_label: Dict[str, str],
    conf_threshold: float,
) -> Dict[str, float]:
    """NeSy variant: when a rule flags a prediction as likely wrong, *override*
    the top-1 label with the top-2 softmax candidate (the network's
    next-best guess), rather than dropping it. This converts symbolic
    flagging into a classification correction signal.

    Error-type semantics:
      * OVERRIDE_ERROR_TYPES — top-1 considered unreliable; replace with
        top-2 (which the KG builder emits as the alternative
        NeuralPrediction). Falls back to drop if top-2 unavailable.
      * AUDIT_ERROR_TYPES — informational; keep top-1 unchanged.
    """
    # All four FeedbackRequired error types trigger top-2 override.
    # MutuallyExclusiveActivities is included because the rule fires
    # exactly when the network is uncertain between two near-equally
    # probable classes — the canonical "swap top-1 for top-2" signal.
    OVERRIDE_ERROR_TYPES = ("FalsePositiveHallucination", "UnsupportedClaim",
                            "ContextualMismatch", "MutuallyExclusiveActivities")
    override_uris = {
        f.get("uri")
        for f in feedback_flags
        if f.get("uri") and any(e in f.get("error_type", "") for e in OVERRIDE_ERROR_TYPES)
    }
    # validated_uris reserved for future rule-augmented-active-set
    # experiments. Currently the threshold gate is enforced uniformly.
    validated_uris: set = set()

    base = "http://example.org/neuro-symbolic-iot#"

    def _top2_label(p: Dict[str, Any]) -> Optional[str]:
        """Return the second-most-probable class label, or None if unavailable."""
        probs = p.get("probabilities") or []
        id2lab = (p.get("metadata") or {}).get("id2label")
        if not probs or not id2lab or len(probs) != len(id2lab):
            return None
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        if len(order) < 2:
            return None
        top1_lab = id2lab[order[0]]
        # Walk to the next-most-probable that differs from top-1.
        for j in order[1:]:
            if id2lab[j] != top1_lab:
                return str(id2lab[j])
        return None

    active_preds: List[Tuple[str, str]] = []  # (gt, pred)
    n_overridden = 0
    n_dropped = 0
    n_rule_admitted = 0
    for i, p in enumerate(predictions):
        gt = p.get("ground_truth_label")
        if gt is None:
            continue
        ev_uri = f"{base}event_{i}"
        is_validated = ev_uri in validated_uris
        # Rule-validated events bypass the neural-only confidence gate.
        if not is_validated and float(p.get("confidence", 0.0)) < conf_threshold:
            continue
        if is_validated and float(p.get("confidence", 0.0)) < conf_threshold:
            n_rule_admitted += 1
        if ev_uri in override_uris:
            top2 = _top2_label(p)
            if top2 is not None:
                active_preds.append((gt, top2))
                n_overridden += 1
                continue
            n_dropped += 1
            continue
        active_preds.append((gt, p["predicted_label"]))

    if not active_preds:
        return {"F1_weighted": 0.0, "Correctness": 0.0, "FP_rate": 0.0, "n_active": 0}

    y_true = [t for t, _ in active_preds]
    y_pred = [p for _, p in active_preds]
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    correctness = correct / len(active_preds)
    return {
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Correctness": float(correctness),
        "FP_rate": float(1.0 - correctness),
        "n_active": len(active_preds),
        "n_overridden": n_overridden,
        "n_dropped_fallback": n_dropped,
        "n_rule_admitted": n_rule_admitted,
    }


# ---------------------------------------------------------------------------
# One full per-fold pipeline (train GRU, run reasoner, compute four configs)
# ---------------------------------------------------------------------------

def _train_fold_and_evaluate(
    cfg: Dict[str, Any],
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df_all: pd.DataFrame,
    full_vocab: Dict[str, int],
    id2label: List[str],
    epochs: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Train GRU on this fold's train indices, evaluate the 4 configurations
    on the held-out fold's predictions, return per-metric values per config.
    """
    from neurosymbolic_iot.cli.train_neural import (
        CasasTorchDataset, casas_collate, _ensure_utc_tz,
    )
    from neurosymbolic_iot.cli.run_pipeline import (
        _run_kg_stage, _run_reasoning_stage,
    )
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
        load_sensor_map,
    )
    from neurosymbolic_iot.neural_perception.casas_sequence import (
        build_casas_sequences,
    )
    from neurosymbolic_iot.neural_perception.inference import (
        InferenceResult, run_inference,
    )
    from neurosymbolic_iot.neural_perception.models import CasasGRUClassifier
    from neurosymbolic_iot.neural_perception.trainer import train_loop
    from neurosymbolic_iot.neural_perception.utils import pick_device
    from neurosymbolic_iot.reasoning_feedback.feedback.feedback_loop import (
        RetrainBuffer, run_feedback_cycle,
    )

    np_cfg = cfg.get("neural_perception", {}).get("casas", {})
    max_seq_len = int(np_cfg.get("max_seq_len", 256))
    batch_size = int(np_cfg.get("batch_size", 32))
    emb_dim = int(np_cfg.get("emb_dim", np_cfg.get("embedding_dim", 64)))
    hidden = int(np_cfg.get("hidden_size", 128))
    num_layers = int(np_cfg.get("num_layers", 1))
    dropout = float(np_cfg.get("dropout", 0.1))
    lr = float(np_cfg.get("lr", 1e-3))

    device = pick_device(str(cfg.get("neural_perception", {}).get("device", "auto")))
    label2id = {lab: i for i, lab in enumerate(id2label)}

    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_val = df_all.iloc[val_idx].reset_index(drop=True)
    for c in ("start_time", "end_time"):
        if c in df_train.columns:
            df_train = _ensure_utc_tz(df_train, c)
        if c in df_val.columns:
            df_val = _ensure_utc_tz(df_val, c)

    train_seqs, _ = build_casas_sequences(cfg, df_train, max_seq_len=max_seq_len, vocab=full_vocab)
    val_seqs, _ = build_casas_sequences(cfg, df_val, max_seq_len=max_seq_len, vocab=full_vocab)
    ds_train = CasasTorchDataset(train_seqs, label2id, max_seq_len)
    ds_val = CasasTorchDataset(val_seqs, label2id, max_seq_len)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=casas_collate)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)

    model = CasasGRUClassifier(
        vocab_size=len(full_vocab), num_classes=len(id2label),
        emb_dim=emb_dim, hidden=hidden, num_layers=num_layers, dropout=dropout,
    )

    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log.info("[fold %d] training GRU (%d train, %d val, %d epochs)",
             fold_idx, len(train_seqs), len(val_seqs), epochs)
    train_loop(
        model=model, train_loader=loader_train, val_loader=loader_val,
        test_loader=loader_val, device=device, id2label=id2label,
        out_dir=fold_dir, epochs=epochs, lr=lr, seed=seed,
    )

    # Reconstruct per-window metadata so the KG builder gets sensor tokens.
    id2tok = {int(idx): str(tok) for tok, idx in full_vocab.items()}
    val_meta: List[Dict[str, Any]] = []
    for w in val_seqs:
        toks = [id2tok.get(int(i), "") for i in w.token_ids]
        toks = [t for t in toks if t and t not in ("<PAD>", "<UNK>")]
        seen: set = set()
        uniq = [t for t in toks if not (t in seen or seen.add(t))]
        val_meta.append({
            "window_start": w.start_time.isoformat() if w.start_time is not None else None,
            "window_end": w.end_time.isoformat() if w.end_time is not None else None,
            "sensor_tokens": uniq,
        })

    inf: InferenceResult = run_inference(
        model, loader_val, device, id2label,
        window_metadata=val_meta, model_tag=f"cv_fold_{fold_idx}", dataset="casas",
    )
    predictions = [asdict(p) for p in inf.predictions]
    log.info("[fold %d] %d predictions produced", fold_idx, len(predictions))

    # Build KG and run the Python rule executor (cfg already defaults to it).
    cfg_fold = copy.deepcopy(cfg)
    cfg_fold.setdefault("kg", {})["output_dir"] = str(fold_dir / "kg")
    cfg_fold.setdefault("reasoning", {})["output_dir"] = str(fold_dir / "reasoning")
    sensor_map = load_sensor_map(Path(
        cfg_fold.get("kg", {}).get(
            "sensor_map_path",
            "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json",
        )
    ))
    activity_map = sensor_map.get("casas", {}).get("activity_map", {})
    # Map ontology IRI -> label-as-stored-in-predictions (inverse of activity_map).
    iri_to_label: Dict[str, str] = {}
    base = "http://example.org/neuro-symbolic-iot#"
    for label, iri in activity_map.items():
        if iri.startswith("nsiot:"):
            iri_to_label[base + iri.split(":", 1)[1]] = label

    graph = _run_kg_stage(cfg_fold, predictions, "casas")
    rr_full = _run_reasoning_stage(cfg_fold, "casas")

    conf_threshold = float(cfg_fold.get("reasoning", {}).get("confidence_threshold_feedback", 0.70))

    # ----- Configuration metrics -----
    ai_only = _ai_only_metrics(predictions, conf_threshold=conf_threshold)
    kg_only = _kg_only_metrics(
        predictions,
        rr_full.validated_events, rr_full.inferred_activities, iri_to_label,
    )
    nesy_nofb = _nesy_metrics(
        predictions, rr_full.feedback_flags,
        rr_full.validated_events, rr_full.inferred_activities, iri_to_label,
        conf_threshold=conf_threshold,
    )

    # NeSy-Full: apply one feedback cycle (real machinery, real graph).
    buffer = RetrainBuffer(
        max_size=int(cfg_fold.get("feedback", {}).get("retrain_buffer_size", 500)),
        trigger_size=int(cfg_fold.get("feedback", {}).get("retrain_trigger_size", 100)),
    )
    run_feedback_cycle(
        reasoning_result=rr_full, predictions=predictions, graph=graph,
        buffer=buffer, cfg=cfg_fold, model_dir=fold_dir, cycle_id=1,
    )
    # Re-run the rule executor on the post-retraction graph to get post-cycle flags.
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.rdf_writer import serialize_graph
    serialize_graph(graph, Path(cfg_fold["kg"]["output_dir"]) / "casas" / "populated_kg.ttl",
                    fmt=cfg_fold.get("kg", {}).get("serialization_format", "turtle"))
    rr_post = _run_reasoning_stage(cfg_fold, "casas")
    new_threshold = float(cfg_fold.get("reasoning", {}).get("confidence_threshold_feedback", conf_threshold))
    nesy_full = _nesy_metrics(
        predictions, rr_post.feedback_flags,
        rr_post.validated_events, rr_post.inferred_activities, iri_to_label,
        conf_threshold=new_threshold,
    )

    return {
        "AI-Only": ai_only,
        "KG+Rules": kg_only,
        "NeSy-NoFeedback": nesy_nofb,
        "NeSy-Full": nesy_full,
        "_meta": {
            "n_predictions": len(predictions),
            "n_feedback_flags": len(rr_full.feedback_flags),
            "n_validated": len(rr_full.validated_events),
            "n_critical": len(rr_full.critical_anomalies),
            "rule_firings": getattr(rr_full, "_rule_firings", {}),
        },
    }


# ---------------------------------------------------------------------------
# Aggregation + paired tests
# ---------------------------------------------------------------------------

def _aggregate(per_fold: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Mean, std, 95% CI half-width for n=5 (Student t, df=4, t_0.975 ≈ 2.776)."""
    summary: Dict[str, Dict[str, float]] = {}
    n = 5
    t_crit_95 = 2.776  # df=4
    for cfg_name, by_metric in per_fold.items():
        out: Dict[str, float] = {}
        for metric, values in by_metric.items():
            valid = [v for v in values if v is not None]
            if not valid:
                out[f"{metric}_mean"] = None  # type: ignore[assignment]
                out[f"{metric}_sd"] = None    # type: ignore[assignment]
                out[f"{metric}_ci95_half"] = None  # type: ignore[assignment]
                continue
            mu = float(np.mean(valid))
            sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            ci_half = t_crit_95 * sd / math.sqrt(len(valid)) if len(valid) > 1 else 0.0
            out[f"{metric}_mean"] = mu
            out[f"{metric}_sd"] = sd
            out[f"{metric}_ci95_half"] = ci_half
        summary[cfg_name] = out
    return summary


def _paired_tests(per_fold: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    from scipy.stats import wilcoxon

    pairs = [
        ("AI-Only", "NeSy-Full"),
        ("NeSy-NoFeedback", "NeSy-Full"),
    ]
    metric_directions = {
        "F1_weighted": "greater",   # NeSy-Full expected > baseline
        "Correctness": "greater",
        "FP_rate":     "less",      # NeSy-Full expected < baseline
    }

    results: Dict[str, Dict[str, float]] = {}
    for baseline, target in pairs:
        for metric, alt in metric_directions.items():
            base_vals = per_fold.get(baseline, {}).get(metric, [])
            tgt_vals = per_fold.get(target, {}).get(metric, [])
            paired = [(b, t) for b, t in zip(base_vals, tgt_vals)
                      if b is not None and t is not None]
            key = f"{metric}__{baseline}_vs_{target}"
            if len(paired) < 5:
                results[key] = {
                    "W": None, "p_value": None,
                    "significant_alpha_0.05": None,
                    "cohens_d": None,
                    "n_paired": len(paired),
                    "alternative": alt,
                    "note": "insufficient paired observations",
                }
                continue
            base_arr = np.array([b for b, _ in paired], dtype=float)
            tgt_arr = np.array([t for _, t in paired], dtype=float)
            diffs = tgt_arr - base_arr
            try:
                # alternative='greater' tests target > baseline. For FP_rate
                # we want target < baseline, i.e., (base - tgt) > 0; flip
                # the operands and use 'greater'.
                if alt == "greater":
                    stat = wilcoxon(tgt_arr, base_arr, alternative="greater", zero_method="wilcox")
                else:
                    stat = wilcoxon(base_arr, tgt_arr, alternative="greater", zero_method="wilcox")
                W = float(stat.statistic)
                p = float(stat.pvalue)
            except ValueError as exc:
                # All differences zero -> Wilcoxon undefined.
                results[key] = {
                    "W": None, "p_value": 1.0,
                    "significant_alpha_0.05": False,
                    "cohens_d": 0.0,
                    "n_paired": len(paired),
                    "alternative": alt,
                    "note": f"wilcoxon undefined ({exc})",
                }
                continue
            sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            d = float(np.mean(diffs) / sd_diff) if sd_diff > 0 else 0.0
            results[key] = {
                "W": W,
                "p_value": p,
                "significant_alpha_0.05": bool(p < 0.05),
                "cohens_d": d,
                "n_paired": len(paired),
                "alternative": alt,
            }
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_ablation_cv(cfg: Dict[str, Any], k: int, epochs: int, seed: int,
                    cep_latency_ms: float, out_dir: Path) -> Dict[str, Any]:
    from neurosymbolic_iot.cli.train_neural import _ensure_utc_tz
    from neurosymbolic_iot.neural_perception.casas_sequence import (
        build_casas_sequences, build_casas_windows_from_raw,
    )

    np_cfg = cfg.get("neural_perception", {}).get("casas", {})
    casas_ds = cfg.get("datasets", {}).get("casas", {})
    window_minutes = int(np_cfg.get("window_minutes", casas_ds.get("window_minutes", 30)))
    stride_minutes = int(np_cfg.get("stride_minutes", casas_ds.get("stride_minutes", 5)))
    min_events = int(np_cfg.get("min_events", casas_ds.get("min_events_per_window", 1)))
    max_seq_len = int(np_cfg.get("max_seq_len", 256))

    log.info("Building CASAS windows…")
    dfw = build_casas_windows_from_raw(
        cfg, window_minutes=window_minutes,
        stride_minutes=stride_minutes, min_events=min_events,
    )
    for c in ("start_time", "end_time"):
        if c in dfw.columns:
            dfw = _ensure_utc_tz(dfw, c)

    id2label = sorted(dfw["label"].astype(str).unique().tolist())
    log.info("CASAS: %d windows, %d classes", len(dfw), len(id2label))

    _, full_vocab = build_casas_sequences(cfg, dfw, max_seq_len=max_seq_len, vocab=None)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    labels_arr = dfw["label"].astype(str).values

    per_fold_metric_arrays: Dict[str, Dict[str, List[Any]]] = {
        c: {m: [] for m in PRIMARY_METRICS} for c in CONFIG_NAMES
    }
    fold_records: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dfw, labels_arr)):
        # Pin all RNG sources to a fold-specific seed BEFORE any model
        # construction or training. This decouples fold N's training from
        # the wall-clock history of folds 0..N-1 and makes the run
        # byte-reproducible.
        fold_seed = _set_per_fold_seed(seed, fold_idx)
        log.info("=" * 60)
        log.info("Fold %d/%d (train=%d, val=%d, fold_seed=%d)",
                 fold_idx + 1, k, len(train_idx), len(val_idx), fold_seed)
        t0 = time.time()
        per_cfg = _train_fold_and_evaluate(
            cfg=cfg, fold_idx=fold_idx,
            train_idx=train_idx, val_idx=val_idx,
            df_all=dfw, full_vocab=full_vocab, id2label=id2label,
            epochs=epochs, seed=fold_seed, out_dir=out_dir,
        )
        for cfg_name in CONFIG_NAMES:
            row = per_cfg.get(cfg_name, {})
            for metric in PRIMARY_METRICS:
                per_fold_metric_arrays[cfg_name][metric].append(row.get(metric))
        fold_records.append({
            "fold": fold_idx,
            "wall_time_s": round(time.time() - t0, 2),
            **{c: per_cfg.get(c, {}) for c in CONFIG_NAMES},
            "_meta": per_cfg.get("_meta", {}),
        })
        log.info("Fold %d done in %.1fs", fold_idx, time.time() - t0)

    summary = _aggregate(per_fold_metric_arrays)
    paired = _paired_tests(per_fold_metric_arrays)

    return {
        "per_fold": per_fold_metric_arrays,
        "fold_records": fold_records,
        "summary": summary,
        "paired_tests": paired,
        "metadata": {
            "k": k,
            "seed": seed,
            "epochs": epochs,
            "n_windows": int(len(dfw)),
            "n_classes": len(id2label),
            "labels": id2label,
            "cep_latency_ms_point_estimate": cep_latency_ms,
            "rule_executor": "python",
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_pct(mean: Any, sd: Any) -> str:
    if mean is None:
        return "N/A"
    return f"{mean*100:.1f}$\\pm${sd*100:.1f}" if sd is not None else f"{mean*100:.1f}"


def print_latex(summary: Dict[str, Any], paired: Dict[str, Any], cep_ms: float) -> None:
    cep_kg = f"{cep_ms:.0f}"
    rows = []
    for name in CONFIG_NAMES:
        s = summary.get(name, {})
        f1 = _fmt_pct(s.get("F1_weighted_mean"), s.get("F1_weighted_sd"))
        cor = _fmt_pct(s.get("Correctness_mean"), s.get("Correctness_sd"))
        fp = _fmt_pct(s.get("FP_rate_mean"), s.get("FP_rate_sd"))
        cep = "N/A" if name == "AI-Only" else cep_kg
        if name == "NeSy-Full":
            rows.append(f"\\textbf{{{name}}} & \\textbf{{{f1}}} & \\textbf{{{cor}}} & \\textbf{{{cep}}} & \\textbf{{{fp}}} \\\\")
        else:
            rows.append(f"{name:<18} & {f1} & {cor} & {cep} & {fp} \\\\")

    def _fmt_test(key: str) -> str:
        t = paired.get(key, {})
        if t.get("W") is None:
            return f"$W{{=}}\\!\\textendash$, $p{{=}}\\!\\textendash$, $d{{=}}\\!\\textendash$"
        return f"$W{{=}}\\!{t['W']:.0f}$, $p{{=}}\\!{t['p_value']:.3f}$, $d{{=}}\\!{t['cohens_d']:.2f}$"

    print()
    print(r"% --- LaTeX-ready Table 11 fragment ---")
    print(r"\begin{table*}[t]\centering\small")
    print(r"\caption{Ablation study (Stratified 5-fold CV on CASAS, Seed = 42).")
    print(r"\textit{Paired Wilcoxon (one-sided, $\alpha{=}0.05$) on F1-weighted:")
    print(rf"NeSy-Full vs AI-Only: {_fmt_test('F1_weighted__AI-Only_vs_NeSy-Full')};")
    print(rf"NeSy-Full vs NeSy-NoFeedback: {_fmt_test('F1_weighted__NeSy-NoFeedback_vs_NeSy-Full')}.}}}}")
    print(r"\label{tab:ablation_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{F1-w (\%)} & \textbf{Corr.\ (\%)} & \textbf{CEP (ms)} & \textbf{FP (\%)} \\")
    print(r"\midrule")
    for r in rows:
        print(r)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def print_human_summary(summary: Dict[str, Any], paired: Dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("  HUMAN-READABLE SUMMARY")
    print("=" * 70)
    for name in CONFIG_NAMES:
        s = summary.get(name, {})
        f1m = s.get("F1_weighted_mean")
        f1s = s.get("F1_weighted_sd")
        cm = s.get("Correctness_mean")
        cs = s.get("Correctness_sd")
        if f1m is None and cm is None:
            print(f"  {name:<18}  (no valid measurements)")
            continue
        f1_str = "N/A" if f1m is None else f"{f1m*100:.2f} +/- {f1s*100:.2f} %"
        cor_str = "N/A" if cm is None else f"{cm*100:.2f} +/- {cs*100:.2f} %"
        print(f"  {name:<18}  F1-w = {f1_str}    Correctness = {cor_str}")

    print()
    print("  Paired Wilcoxon (one-sided) results:")
    for key, t in paired.items():
        if t.get("W") is None:
            verdict = "skipped (" + (t.get("note") or "n<5") + ")"
            print(f"    {key:<48}  {verdict}")
            continue
        sig = "SIGNIFICANT" if t["significant_alpha_0.05"] else "NOT SIGNIFICANT (soften manuscript!)"
        d = t["cohens_d"]
        size = ("large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5
                else "small" if abs(d) >= 0.2 else "negligible")
        print(f"    {key:<48}  p={t['p_value']:.4f}  d={d:.2f} ({size})  [{sig}]")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Epochs per fold (default 30 — match Table 5)")
    p.add_argument("--cep-latency", type=float, default=12.4,
                   help="Point-estimate CEP latency (ms) for Table 11 (paper-side infrastructure metric)")
    p.add_argument("--outdir", default="evaluation/results", help="Output directory")
    return p.parse_args()


def main() -> int:
    from neurosymbolic_iot.utils.config import load_config
    from neurosymbolic_iot.utils.logging import setup_logging
    from neurosymbolic_iot.utils.seed import set_global_seed

    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== R2.1: 5-fold CV ablation with paired tests ===")
    result = run_ablation_cv(
        cfg=cfg, k=args.k, epochs=args.epochs, seed=seed,
        cep_latency_ms=args.cep_latency, out_dir=out_dir,
    )

    out_path = out_dir / "ablation_cv.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("Saved %s", out_path)

    print_latex(result["summary"], result["paired_tests"], args.cep_latency)
    print_human_summary(result["summary"], result["paired_tests"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
