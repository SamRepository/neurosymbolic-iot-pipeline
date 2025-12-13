from __future__ import annotations

import argparse
import logging

from neurosymbolic_iot.data_processing.preprocess_casas import preprocess_casas
from neurosymbolic_iot.data_processing.preprocess_sphere import preprocess_sphere
from neurosymbolic_iot.utils.config import ensure_dirs, load_config
from neurosymbolic_iot.utils.logging import setup_logging
from neurosymbolic_iot.utils.seed import set_global_seed

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 preprocessing for CASAS and SPHERE datasets.")
    p.add_argument("--config", required=True, help="Path to YAML config (supports inherits).")
    p.add_argument("--dataset", required=True, choices=["casas", "sphere", "all"], help="Which dataset to preprocess.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_dirs(cfg)
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"))

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    if args.dataset in ("casas", "all"):
        preprocess_casas(cfg)

    if args.dataset in ("sphere", "all"):
        preprocess_sphere(cfg)


if __name__ == "__main__":
    main()
