from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict b into dict a (b overrides a).
    """
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_inherit_path(current_cfg_path: Path, parent_spec: str) -> Path:
    """
    Resolve an inherited config path robustly.

    Tries:
      1) relative to the current config file directory
      2) relative to repo CWD
      3) relative to parent of config dir (useful if parent_spec includes 'config/')
    """
    p = Path(parent_spec)
    if p.is_absolute():
        return p

    candidates = [
        current_cfg_path.parent / p,
        Path.cwd() / p,
        current_cfg_path.parent.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return c

    # Default to the most sensible path so the error message is useful.
    return candidates[0]


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML config file with optional inheritance.

    Supports:
      inherits: "base.yaml"
      inherits:
        - "base.yaml"
        - "other.yaml"
    """
    path = Path(path)
    cfg = load_yaml(path)

    merged: Dict[str, Any] = {}

    parents = cfg.get("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    elif parents is None:
        parents = []
    elif not isinstance(parents, (list, tuple)):
        raise TypeError(f"'inherits' must be a string or a list, got: {type(parents)}")

    for parent_spec in parents:
        parent_path = _resolve_inherit_path(path, str(parent_spec))
        merged = _deep_merge(merged, load_config(parent_path))

    merged = _deep_merge(merged, cfg)
    return merged
