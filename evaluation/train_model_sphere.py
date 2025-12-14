from __future__ import annotations

import argparse
import logging
from pathlib import Path

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

from evaluation.common import load_parquet, safe_train_val_test_split, train_and_evaluate

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline activity classifier on SPHERE windows.")
    parser.add_argument("--config", required=True, help="Path to YAML config (supports inherits).")
    parser.add_argument("--outdir", default="outputs/evaluation", help="Where to write models/metrics.")
    parser.add_argument("--model", default=None, help="Model override: rf/random_forest.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    sphere_parquet = Path(cfg["output"]["sphere_windows"])
    df = load_parquet(sphere_parquet)

    if "split" in df.columns:
        df_train = df[df["split"] == "train"].reset_index(drop=True)
        df_val = df[df["split"] == "val"].reset_index(drop=True)
        df_test = df[df["split"] == "test"].reset_index(drop=True)
    else:
        ds_cfg = cfg.get("datasets", {}).get("sphere", {})
        train = float(ds_cfg.get("train_ratio", 0.7))
        val = float(ds_cfg.get("val_ratio", 0.15))
        test = float(ds_cfg.get("test_ratio", 0.15))
        splits = safe_train_val_test_split(df, train=train, val=val, test=test, seed=seed)
        df_train, df_val, df_test = splits["train"], splits["val"], splits["test"]

    art = train_and_evaluate(
        dataset="sphere",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        out_dir=args.outdir,
        seed=seed,
        model_name=args.model,
    )
    log.info("SPHERE training complete. Metrics: %s", art.metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())