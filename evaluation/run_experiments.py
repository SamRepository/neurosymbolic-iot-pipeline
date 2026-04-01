from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, List

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

from evaluation.common import load_parquet, safe_train_val_test_split, save_json, train_and_evaluate

log = logging.getLogger(__name__)


def _timestamp_tag() -> str:
  #  return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
  return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def _train_casas(cfg: Dict[str, Any], outdir: Path, model_override: str | None) -> Dict[str, Any]:
    df = load_parquet(cfg["output"]["casas_windows"])
    ds_cfg = cfg.get("datasets", {}).get("casas", {})
    train = float(ds_cfg.get("train_ratio", 0.7))
    val = float(ds_cfg.get("val_ratio", 0.15))
    test = float(ds_cfg.get("test_ratio", 0.15))

    splits = safe_train_val_test_split(
        df,
        train=train,
        val=val,
        test=test,
        seed=int(cfg.get("project", {}).get("seed", 42)),
        stratify_col="label",
    )

    art = train_and_evaluate(
        dataset="casas",
        df_train=splits["train"],
        df_val=splits["val"],
        df_test=splits["test"],
        out_dir=outdir,
        seed=int(cfg.get("project", {}).get("seed", 42)),
        model_name=model_override,
    )
    return {"dataset": "casas", "artifacts": art.__dict__}


def _train_sphere(cfg: Dict[str, Any], outdir: Path, model_override: str | None) -> Dict[str, Any]:
    df = load_parquet(cfg["output"]["sphere_windows"])

    if "split" in df.columns:
        df_train = df[df["split"] == "train"].reset_index(drop=True)
        df_val = df[df["split"] == "val"].reset_index(drop=True)
        df_test = df[df["split"] == "test"].reset_index(drop=True)
    else:
        ds_cfg = cfg.get("datasets", {}).get("sphere", {})
        train = float(ds_cfg.get("train_ratio", 0.7))
        val = float(ds_cfg.get("val_ratio", 0.15))
        test = float(ds_cfg.get("test_ratio", 0.15))
        splits = safe_train_val_test_split(
            df,
            train=train,
            val=val,
            test=test,
            seed=int(cfg.get("project", {}).get("seed", 42)),
            stratify_col="label",
        )
        df_train, df_val, df_test = splits["train"], splits["val"], splits["test"]

    art = train_and_evaluate(
        dataset="sphere",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        out_dir=outdir,
        seed=int(cfg.get("project", {}).get("seed", 42)),
        model_name=model_override,
    )
    return {"dataset": "sphere", "artifacts": art.__dict__}



# ---------------------------------------------------------------------------
# Pipeline / Ablation mode (Table 6 from paper)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "ai_only": "config/ai_only.yaml",
    "kg_only": "config/kg_only.yaml",
    "ns_nofeedback": "config/ns_nofeedback.yaml",
    "ns_full": "config/ns_full.yaml",
}


def _run_pipeline_mode(cfg: Dict[str, Any], args: argparse.Namespace, tag: str, outdir: Path) -> int:
    """Run full neuro-symbolic pipeline or ablation study (Table 6)."""
    from neurosymbolic_iot.cli.run_pipeline import run_pipeline

    model_dir = Path(args.model_dir) if args.model_dir else None
    if model_dir is None or not model_dir.exists():
        log.error("--model-dir is required for pipeline/ablation mode and must exist.")
        return 1

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    all_results: List[Dict[str, Any]] = []

    if args.mode == "ablation":
        # Run all 4 ablation configs for each dataset
        for ablation_name, config_path in ABLATION_CONFIGS.items():
            config_file = Path(config_path)
            if not config_file.exists():
                log.warning("Ablation config %s not found — skipping.", config_path)
                continue

            ablation_cfg = load_config(str(config_file))
            for ds in datasets:
                log.info("=== Ablation: %s / %s ===", ablation_name, ds)
                result = run_pipeline(
                    cfg=ablation_cfg,
                    dataset=ds,
                    task="activity",
                    model_dir=model_dir,
                    tag=f"{tag}/{ablation_name}",
                )
                all_results.append({
                    "ablation": ablation_name,
                    "dataset": ds,
                    "num_predictions": result.num_predictions,
                    "kg_triples": result.kg_triples,
                    "inferred_triples": result.inferred_triples,
                    "validated_events": result.validated_events,
                    "critical_anomalies": result.critical_anomalies,
                    "behavioral_anomalies": result.behavioral_anomalies,
                    "feedback_flags": result.feedback_flags,
                    "feedback_cycles": result.feedback_cycles_run,
                    "final_fp_retracted": result.final_fp_retracted,
                    "total_seconds": round(result.timings.total_seconds, 3),
                })
    else:
        # Single pipeline run with the provided config
        for ds in datasets:
            log.info("=== Pipeline: %s ===", ds)
            result = run_pipeline(
                cfg=cfg,
                dataset=ds,
                task=args.datasets.split(",")[0] if args.datasets else "activity",
                model_dir=model_dir,
                tag=tag,
            )
            all_results.append({
                "dataset": ds,
                "num_predictions": result.num_predictions,
                "kg_triples": result.kg_triples,
                "validated_events": result.validated_events,
                "feedback_cycles": result.feedback_cycles_run,
                "total_seconds": round(result.timings.total_seconds, 3),
            })

    # Save ablation/pipeline summary
    summary = {
        "mode": args.mode,
        "tag": tag,
        "results": all_results,
    }
    save_json(outdir / f"{args.mode}_results.json", summary)
    log.info("Saved %s results to %s", args.mode, outdir / f"{args.mode}_results.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 baselines (AI-only activity recognition).")
    parser.add_argument("--config", required=True, help="Path to YAML config (supports inherits).")
    parser.add_argument("--datasets", default="casas,sphere", help="Comma-separated list: casas,sphere")
    parser.add_argument("--outdir", default="outputs/evaluation", help="Output directory for models/metrics.")
    parser.add_argument("--tag", default=None, help="Optional tag. Default = UTC timestamp.")
    parser.add_argument("--model", default=None, help="Model override (rf/random_forest).")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "pipeline", "ablation"],
        help="baseline: scikit-learn baselines (default). pipeline: full neuro-symbolic pipeline. ablation: run all 4 ablation configs.",
    )
    parser.add_argument("--model-dir", default=None, help="Trained model dir (required for pipeline/ablation mode).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    tag = args.tag or _timestamp_tag()
    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("pipeline", "ablation"):
        return _run_pipeline_mode(cfg, args, tag, outdir)

    requested = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    results: List[Dict[str, Any]] = []

    missing = []
    if "casas" in requested and not Path(cfg["output"]["casas_windows"]).exists():
        missing.append(cfg["output"]["casas_windows"])
    if "sphere" in requested and not Path(cfg["output"]["sphere_windows"]).exists():
        missing.append(cfg["output"]["sphere_windows"])
    if missing:
        raise FileNotFoundError(
            "Missing processed parquet file(s). Run preprocessing first.\n"
            f"Missing: {missing}"
        )

    if "casas" in requested:
        results.append(_train_casas(cfg, outdir, args.model))
    if "sphere" in requested:
        results.append(_train_sphere(cfg, outdir, args.model))

    summary = {"tag": tag, "config": str(Path(args.config)), "seed": seed, "datasets": requested, "results": results}
    save_json(outdir / "run_summary.json", summary)
    log.info("Saved run summary: %s", (outdir / "run_summary.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
