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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 baselines (AI-only activity recognition).")
    parser.add_argument("--config", required=True, help="Path to YAML config (supports inherits).")
    parser.add_argument("--datasets", default="casas,sphere", help="Comma-separated list: casas,sphere")
    parser.add_argument("--outdir", default="outputs/evaluation", help="Output directory for models/metrics.")
    parser.add_argument("--tag", default=None, help="Optional tag. Default = UTC timestamp.")
    parser.add_argument("--model", default=None, help="Model override (rf/random_forest).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    tag = args.tag or _timestamp_tag()
    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

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
