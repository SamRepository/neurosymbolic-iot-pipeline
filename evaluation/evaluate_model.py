from __future__ import annotations

import argparse
import logging
from pathlib import Path

from neurosymbolic_iot.utils.config import load_config
from neurosymbolic_iot.utils.logging import setup_logging

from evaluation.common import (
    compute_metrics,
    load_artifacts,
    load_parquet,
    make_xy,
    save_confusion_csv,
    save_json,
)

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on processed windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["casas", "sphere"])
    parser.add_argument("--artifact_dir", default="outputs/evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    model, le, feature_cols = load_artifacts(args.artifact_dir, args.dataset)

    parquet_path = Path(cfg["output"][f"{args.dataset}_windows"])
    df = load_parquet(parquet_path)

    if args.dataset == "sphere" and "split" in df.columns:
        df_eval = df[df["split"] == args.split].reset_index(drop=True)
    else:
        df_eval = df.copy()

    missing = [c for c in feature_cols if c not in df_eval.columns]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} feature columns (example: {missing[:10]}). "
            "You likely trained on a different preprocessing output."
        )

    X, y_true = make_xy(df_eval, feature_cols, label_col="label")
    y_pred_enc = model.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)

    metrics = {
        "dataset": args.dataset,
        "split": args.split if args.dataset == "sphere" else "all",
        "n_rows": int(len(df_eval)),
        "metrics": compute_metrics(y_true, y_pred),
    }

    out_dir = Path(args.artifact_dir) / args.dataset
    out_json = Path(args.out) if args.out else (out_dir / f"eval_{metrics['split']}.json")
    save_json(out_json, metrics)

    out_cm = out_dir / f"confusion_eval_{metrics['split']}.csv"
    save_confusion_csv(y_true, y_pred, labels=le.classes_, out_csv=out_cm)

    log.info("Saved evaluation JSON: %s", out_json.as_posix())
    log.info("Saved confusion matrix: %s", out_cm.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
