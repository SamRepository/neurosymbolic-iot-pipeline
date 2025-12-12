from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a config file that may declare `inherits: [...]` to compose configs."""
    path = Path(path)
    cfg = load_yaml(path)

    inherits = cfg.get("inherits", [])
    merged: Dict[str, Any] = {}

    # Support relative paths in inherits
    for parent in inherits:
        parent_path = (path.parent / parent).resolve()
        merged = _deep_merge(merged, load_config(parent_path))

    merged = _deep_merge(merged, cfg)
    # `inherits` is an implementation detail; keep it out of runtime config
    merged.pop("inherits", None)
    return merged


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    processed_root = Path(cfg["paths"]["processed_root"])
    logs_root = Path(cfg["paths"]["logs_root"])
    processed_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
