from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

log = logging.getLogger(__name__)


@dataclass
class StageTimings:
    """Tracks wall-clock time for each pipeline stage (Tables 3-5)."""

    inference_seconds: float = 0.0
    kg_build_seconds: float = 0.0
    kg_serialize_seconds: float = 0.0
    reasoning_seconds: float = 0.0
    feedback_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class PipelineResult:
    """Aggregated output from a full pipeline run."""

    dataset: str = ""
    task: str = ""
    model_dir: str = ""
    tag: str = ""
    num_predictions: int = 0
    kg_triples: int = 0
    inferred_triples: int = 0
    validated_events: int = 0
    critical_anomalies: int = 0
    behavioral_anomalies: int = 0
    feedback_flags: int = 0
    feedback_cycles_run: int = 0
    final_fp_retracted: int = 0
    timings: StageTimings = field(default_factory=StageTimings)
    feedback_cycle_details: List[Dict[str, Any]] = field(default_factory=list)


def _utc_tag() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 1: Neural Inference
# ---------------------------------------------------------------------------

def _run_neural_stage(
    cfg: Dict[str, Any],
    dataset: str,
    task: str,
    model_dir: Path,
    device: str,
) -> List[Dict[str, Any]]:
    """Load trained model, run inference, return list of prediction dicts."""
    from neurosymbolic_iot.neural_perception.inference import (
        InferenceResult,
        load_trained_model,
        run_inference,
        save_predictions,
    )

    model, id2label, label2id, task_info = load_trained_model(model_dir, device, dataset, cfg)

    # Rebuild the test DataLoader from the same data pipeline used during training.
    # Also recover per-window metadata (sensor tokens, timestamps, active PIRs)
    # so the KG builder can assert sensor states + room locations needed by the
    # SWRL rule preconditions.
    loader, window_metadata = _build_test_loader(cfg, dataset, task, model_dir, label2id)

    result = run_inference(
        model,
        loader,
        device,
        id2label,
        window_metadata=window_metadata,
        model_tag=model_dir.name,
        dataset=dataset,
    )

    # Save predictions
    out_path = Path(cfg.get("neural_perception", {}).get("output_dir", "outputs/neural_perception"))
    save_predictions(result, out_path / dataset / "predictions.json")

    # Convert to plain dicts for downstream stages
    return [asdict(p) for p in result.predictions]


def _build_test_loader(
    cfg: Dict[str, Any],
    dataset: str,
    task: str,
    model_dir: Path,
    label2id: Dict[str, int],
) -> "tuple[DataLoader, List[Dict[str, Any]]]":
    """Reconstruct the test DataLoader plus per-window metadata for inference.

    The metadata list is aligned with the loader's iteration order. Each entry
    contains ``window_start`` / ``window_end`` (ISO-8601), and:
      * for CASAS: ``sensor_tokens`` recovered from the trained vocab
      * for SPHERE: ``active_pirs`` derived from PIR observations within the
        window's time interval
    These fields drive the sensor-state and room-location assertions used by
    the SWRL rules in the symbolic-reasoning stage.
    """
    from neurosymbolic_iot.neural_perception.utils import safe_split_df

    seed = int(cfg.get("project", {}).get("seed", 42))

    if dataset == "casas":
        from neurosymbolic_iot.neural_perception.casas_sequence import (
            build_casas_sequences,
            build_casas_windows_from_raw,
        )
        from neurosymbolic_iot.cli.train_neural import (
            CasasTorchDataset,
            add_transition_labels,
            casas_collate,
        )

        np_cfg = cfg.get("neural_perception", {}).get("casas", {})
        casas_ds = cfg.get("datasets", {}).get("casas", {})
        window_minutes = int(np_cfg.get("window_minutes", casas_ds.get("window_minutes", 30)))
        stride_minutes = int(np_cfg.get("stride_minutes", casas_ds.get("stride_minutes", 5)))
        min_events = int(np_cfg.get("min_events", casas_ds.get("min_events_per_window", 1)))
        max_seq_len = int(np_cfg.get("max_seq_len", 256))
        batch_size = int(np_cfg.get("batch_size", 32))
        train_ratio = float(np_cfg.get("train_ratio", 0.7))
        val_ratio = float(np_cfg.get("val_ratio", 0.15))
        test_ratio = float(np_cfg.get("test_ratio", 0.15))

        dfw = build_casas_windows_from_raw(
            cfg,
            window_minutes=window_minutes,
            stride_minutes=stride_minutes,
            min_events=min_events,
        )

        if task == "transition":
            dfw = add_transition_labels(dfw)
            dfw["label"] = dfw["label_transition"].astype(str)

        splits = safe_split_df(dfw, train=train_ratio, val=val_ratio, test=test_ratio, seed=seed, stratify_col="label")

        # Load vocab from model dir
        import json as _json
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab = _json.load(f)

        test_seqs, _ = build_casas_sequences(cfg, splits.test, max_seq_len=max_seq_len, vocab=vocab)
        ds = CasasTorchDataset(test_seqs, label2id, max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=casas_collate)

        # Per-window metadata aligned with test_seqs iteration order.
        id2tok = {int(idx): str(tok) for tok, idx in vocab.items()}
        meta: List[Dict[str, Any]] = []
        for w in test_seqs:
            tokens = [id2tok.get(int(i), "") for i in w.token_ids]
            tokens = [t for t in tokens if t and t not in ("<PAD>", "<UNK>")]
            # Deduplicate while preserving order for KG triple compactness.
            seen: set = set()
            uniq_tokens = [t for t in tokens if not (t in seen or seen.add(t))]
            meta.append({
                "window_start": w.start_time.isoformat() if w.start_time is not None else None,
                "window_end": w.end_time.isoformat() if w.end_time is not None else None,
                "sensor_tokens": uniq_tokens,
            })
        return loader, meta

    elif dataset == "sphere":
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
        train_ratio = float(np_cfg.get("train_ratio", 0.7))
        val_ratio = float(np_cfg.get("val_ratio", 0.15))
        test_ratio = float(np_cfg.get("test_ratio", 0.15))

        import pandas as pd
        labeled = build_sphere_labeled_stream(cfg)
        windows = build_sphere_windows(labeled, window_seconds=window_seconds, stride_seconds=stride_seconds, min_rows=min_rows, seq_len=seq_len)
        dfw = pd.DataFrame([{"idx": i, "label": str(w.label)} for i, w in enumerate(windows)])
        splits = safe_split_df(dfw, train=train_ratio, val=val_ratio, test=test_ratio, seed=seed, stratify_col="label")
        test_w = [windows[int(i)] for i in splits.test["idx"].tolist()]
        ds = SphereTorchDataset(test_w, label2id)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # SPHERE per-window metadata: timestamps + active_pirs derived from
        # the raw PIR observation file, joined to each window's [start, end].
        meta: List[Dict[str, Any]] = []
        active_pirs_by_window = _sphere_active_pirs(cfg, test_w)
        for i, w in enumerate(test_w):
            meta.append({
                "window_start": w.start_time.isoformat() if w.start_time is not None else None,
                "window_end": w.end_time.isoformat() if w.end_time is not None else None,
                "active_pirs": active_pirs_by_window[i],
            })
        return loader, meta

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _sphere_active_pirs(cfg: Dict[str, Any], windows: List[Any]) -> List[List[str]]:
    """For each SPHERE window, return the list of PIR room names whose value
    is non-zero anywhere within the window. Returns ``[[]] * len(windows)``
    if the PIR file is missing — the SWRL rules requiring sensor-state
    grounding simply will not fire for those windows.
    """
    import pandas as pd

    raw_dir = Path(cfg.get("datasets", {}).get("sphere", {}).get("raw_dir", "data/raw/sphere"))
    pir_path = raw_dir / "pir.csv"
    if not pir_path.exists():
        log.info("SPHERE pir.csv not found at %s — skipping active_pirs metadata.", pir_path)
        return [[] for _ in windows]

    try:
        pir = pd.read_csv(pir_path)
    except Exception as exc:
        log.warning("Failed to read %s: %s", pir_path, exc)
        return [[] for _ in windows]

    ts_col = next((c for c in ("timestamp", "t", "time") if c in pir.columns), None)
    if ts_col is None:
        log.warning("SPHERE pir.csv has no timestamp column — skipping active_pirs.")
        return [[] for _ in windows]
    pir[ts_col] = pd.to_datetime(pir[ts_col], errors="coerce", utc=False)
    pir = pir.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    pir_cols = [c for c in pir.columns if c != ts_col]
    out: List[List[str]] = []
    for w in windows:
        t0, t1 = w.start_time, w.end_time
        sub = pir[(pir[ts_col] >= t0) & (pir[ts_col] < t1)]
        if sub.empty:
            out.append([])
            continue
        active = [c for c in pir_cols if (sub[c].fillna(0).astype(float) > 0).any()]
        out.append(active)
    return out


# ---------------------------------------------------------------------------
# Stage 2: Knowledge Graph Construction
# ---------------------------------------------------------------------------

def _run_kg_stage(
    cfg: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    dataset: str,
) -> Any:
    """Build RDF graph from predictions and serialize to disk."""
    from rdflib import Graph

    from neurosymbolic_iot.kg_semantic_layer.kg_builder.kg_federation_loader import (
        build_kg_from_predictions,
        load_sensor_map,
    )
    from neurosymbolic_iot.kg_semantic_layer.kg_builder.rdf_writer import (
        push_to_graphdb,
        serialize_graph,
    )

    kg_cfg = cfg.get("kg", {})
    sensor_map_path = Path(kg_cfg.get("sensor_map_path", "neurosymbolic_iot/kg_semantic_layer/ontology/sensor_map.json"))
    ontology_path = Path(kg_cfg.get("ontology_path", ""))
    out_dir = Path(kg_cfg.get("output_dir", "outputs/kg")) / dataset
    fmt = kg_cfg.get("serialization_format", "turtle")

    sensor_map = load_sensor_map(sensor_map_path)

    g = build_kg_from_predictions(
        predictions,
        sensor_map,
        dataset,
        ontology_path=ontology_path if ontology_path.exists() else None,
    )

    # Serialize
    out_file = out_dir / "populated_kg.ttl"
    serialize_graph(g, out_file, fmt=fmt)

    # Optional GraphDB push
    gdb = kg_cfg.get("graphdb", {})
    if gdb.get("enabled", False):
        push_to_graphdb(
            g,
            endpoint=gdb["endpoint"],
            repository=gdb["repository"],
            named_graph=f"http://example.org/neuro-symbolic-iot/{dataset}",
            auth=gdb.get("auth"),
        )

    return g


# ---------------------------------------------------------------------------
# Stage 3: Symbolic Reasoning
# ---------------------------------------------------------------------------

def _run_reasoning_stage(
    cfg: Dict[str, Any],
    dataset: str,
) -> Any:
    """Run OWL/SWRL reasoning over the populated KG."""
    from neurosymbolic_iot.reasoning_feedback.reasoning.symbolic_reasoner import (
        ReasoningResult,
        run_reasoning,
    )

    kg_cfg = cfg.get("kg", {})
    reasoning_cfg = cfg.get("reasoning", {})

    ontology_path = Path(kg_cfg.get("ontology_path", ""))
    populated_kg = Path(kg_cfg.get("output_dir", "outputs/kg")) / dataset / "populated_kg.ttl"
    out_dir = Path(reasoning_cfg.get("output_dir", "outputs/reasoning")) / dataset

    result = run_reasoning(ontology_path, populated_kg, cfg=cfg)

    # Save reasoning result
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "reasoning_result.json", {
        "validated_events": result.validated_events,
        "critical_anomalies": result.critical_anomalies,
        "behavioral_anomalies": result.behavioral_anomalies,
        "feedback_flags": result.feedback_flags,
        "inferred_activities": result.inferred_activities,
        "inferred_locations": result.inferred_locations,
        "inconsistencies": result.inconsistencies,
        "num_inferred_triples": result.num_inferred_triples,
        "reasoning_time_seconds": result.reasoning_time_seconds,
    })

    return result


# ---------------------------------------------------------------------------
# Stage 4: Feedback Loop
# ---------------------------------------------------------------------------

def _run_feedback_stage(
    cfg: Dict[str, Any],
    reasoning_result: Any,
    predictions: List[Dict[str, Any]],
    graph: Any,
    dataset: str,
    model_dir: Path,
) -> List[Dict[str, Any]]:
    """Run N feedback cycles. Returns list of per-cycle summaries."""
    from neurosymbolic_iot.reasoning_feedback.feedback.feedback_loop import (
        FeedbackCycle,
        RetrainBuffer,
        run_feedback_cycle,
    )

    fb_cfg = cfg.get("feedback", {})
    max_cycles = int(fb_cfg.get("max_cycles", 5))
    buffer_size = int(fb_cfg.get("retrain_buffer_size", 500))
    trigger_size = int(fb_cfg.get("retrain_trigger_size", 100))
    out_dir = Path(fb_cfg.get("output_dir", "outputs/feedback")) / dataset

    buffer = RetrainBuffer(max_size=buffer_size, trigger_size=trigger_size)
    cycle_details: List[Dict[str, Any]] = []

    for i in range(max_cycles):
        cycle = run_feedback_cycle(
            reasoning_result=reasoning_result,
            predictions=predictions,
            graph=graph,
            buffer=buffer,
            cfg=cfg,
            model_dir=model_dir,
            cycle_id=i,
        )
        detail = asdict(cycle)
        cycle_details.append(detail)
        log.info(
            "Feedback cycle %d/%d: contradictions=%d, retracted=%d, buffered=%d",
            i + 1, max_cycles,
            cycle.contradictions_detected,
            cycle.false_positives_retracted,
            cycle.samples_buffered_for_retrain,
        )

        # Early stop if no contradictions found
        if cycle.contradictions_detected == 0:
            log.info("No contradictions in cycle %d — converged, stopping early.", i)
            break

    # Save feedback results
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "feedback_cycles.json", cycle_details)

    return cycle_details


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: Dict[str, Any],
    dataset: str,
    task: str,
    model_dir: Path,
    tag: str,
    feedback_cycles: Optional[int] = None,
) -> PipelineResult:
    """
    End-to-end neuro-symbolic pipeline (Algorithms 1 + 2 from paper).

    Stages (each gated by pipeline.enable_* flags):
      1. Neural inference → softmax predictions
      2. KG construction → RDF graph from predictions
      3. Symbolic reasoning → OWL/SWRL via HermiT
      4. Feedback loop → contradiction detection, retraction, threshold update
    """
    pipeline_cfg = cfg.get("pipeline", {})
    result = PipelineResult(dataset=dataset, task=task, model_dir=str(model_dir), tag=tag)
    timings = StageTimings()

    if feedback_cycles is not None:
        cfg.setdefault("feedback", {})["max_cycles"] = feedback_cycles

    device = str(cfg.get("neural_perception", {}).get("device", "auto"))
    from neurosymbolic_iot.neural_perception.utils import pick_device
    device = pick_device(device)

    t_total = time.time()
    predictions: List[Dict[str, Any]] = []
    graph = None
    reasoning_result = None

    # Stage 1: Neural Inference
    if pipeline_cfg.get("enable_neural", True):
        log.info("=== Stage 1: Neural Inference ===")
        t0 = time.time()
        predictions = _run_neural_stage(cfg, dataset, task, model_dir, device)
        timings.inference_seconds = time.time() - t0
        result.num_predictions = len(predictions)
        log.info("Stage 1 done: %d predictions in %.2fs", len(predictions), timings.inference_seconds)

    # Stage 2: KG Construction
    if pipeline_cfg.get("enable_kg", True) and predictions:
        log.info("=== Stage 2: Knowledge Graph Construction ===")
        t0 = time.time()
        graph = _run_kg_stage(cfg, predictions, dataset)
        timings.kg_build_seconds = time.time() - t0
        result.kg_triples = len(graph)
        log.info("Stage 2 done: %d triples in %.2fs", result.kg_triples, timings.kg_build_seconds)

    # Stage 3: Symbolic Reasoning
    if pipeline_cfg.get("enable_symbolic", True) and graph is not None:
        log.info("=== Stage 3: Symbolic Reasoning ===")
        t0 = time.time()
        reasoning_result = _run_reasoning_stage(cfg, dataset)
        timings.reasoning_seconds = time.time() - t0
        result.inferred_triples = reasoning_result.num_inferred_triples
        result.validated_events = len(reasoning_result.validated_events)
        result.critical_anomalies = len(reasoning_result.critical_anomalies)
        result.behavioral_anomalies = len(reasoning_result.behavioral_anomalies)
        result.feedback_flags = len(reasoning_result.feedback_flags)
        log.info(
            "Stage 3 done: %d inferred triples, %d validated, %d critical, %d behavioral, %d feedback in %.2fs",
            result.inferred_triples, result.validated_events, result.critical_anomalies,
            result.behavioral_anomalies, result.feedback_flags, timings.reasoning_seconds,
        )

    # Stage 4: Feedback Loop
    if pipeline_cfg.get("enable_feedback", True) and reasoning_result is not None and predictions:
        log.info("=== Stage 4: Feedback Loop ===")
        t0 = time.time()
        cycle_details = _run_feedback_stage(
            cfg, reasoning_result, predictions, graph, dataset, model_dir,
        )
        timings.feedback_seconds = time.time() - t0
        result.feedback_cycles_run = len(cycle_details)
        result.feedback_cycle_details = cycle_details
        result.final_fp_retracted = sum(c.get("false_positives_retracted", 0) for c in cycle_details)
        log.info("Stage 4 done: %d cycles in %.2fs", result.feedback_cycles_run, timings.feedback_seconds)

    timings.total_seconds = time.time() - t_total
    result.timings = timings

    # Save overall pipeline result
    out_dir = Path(cfg.get("output", {}).get("pipeline_dir", "outputs/pipeline")) / dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "pipeline_result.json", asdict(result))
    log.info("Pipeline complete in %.2fs — results saved to %s", timings.total_seconds, out_dir)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full neuro-symbolic IoT pipeline (inference → KG → reasoning → feedback).",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. config/ns_full.yaml)")
    parser.add_argument("--dataset", required=True, choices=["casas", "sphere"], help="Dataset to process")
    parser.add_argument("--task", default="activity", choices=["activity", "transition"], help="CASAS task type")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory (contains model.pth)")
    parser.add_argument("--tag", default=None, help="Run tag (default: UTC timestamp)")
    parser.add_argument("--feedback-cycles", type=int, default=None, help="Override feedback.max_cycles from config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    tag = args.tag or _utc_tag()
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        log.error("Model directory does not exist: %s", model_dir)
        return 1

    log.info("Starting pipeline: dataset=%s, task=%s, model=%s, tag=%s", args.dataset, args.task, model_dir, tag)

    result = run_pipeline(
        cfg=cfg,
        dataset=args.dataset,
        task=args.task,
        model_dir=model_dir,
        tag=tag,
        feedback_cycles=args.feedback_cycles,
    )

    log.info(
        "Pipeline summary: %d predictions, %d KG triples, %d inferred, "
        "%d validated, %d anomalies, %d feedback cycles (%.1fs total)",
        result.num_predictions, result.kg_triples, result.inferred_triples,
        result.validated_events, result.critical_anomalies + result.behavioral_anomalies,
        result.feedback_cycles_run, result.timings.total_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
