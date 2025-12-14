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
    parser = argparse.ArgumentParser(description="Train a baseline activity classifier on CASAS windows.")
    parser.add_argument("--config", required=True, help="Path to YAML config (supports inherits).")
    parser.add_argument("--outdir", default="outputs/evaluation", help="Where to write models/metrics.")
    parser.add_argument("--model", default=None, help="Model override: rf/random_forest.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    casas_parquet = Path(cfg["output"]["casas_windows"])
    df = load_parquet(casas_parquet)

    ds_cfg = cfg.get("datasets", {}).get("casas", {})
    train = float(ds_cfg.get("train_ratio", 0.7))
    val = float(ds_cfg.get("val_ratio", 0.15))
    test = float(ds_cfg.get("test_ratio", 0.15))

    splits = safe_train_val_test_split(df, train=train, val=val, test=test, seed=seed)

    art = train_and_evaluate(
        dataset="casas",
        df_train=splits["train"],
        df_val=splits["val"],
        df_test=splits["test"],
        out_dir=args.outdir,
        seed=seed,
        model_name=args.model,
    )
    log.info("CASAS training complete. Metrics: %s", art.metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
