from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges override into base (override wins).
    """
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, Mapping)
        ):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML config file with optional inheritance via:
      extends: "base.yaml"
    or
      extends:
        - "base.yaml"
        - "other.yaml"

    Paths in 'extends' are resolved relative to the current config file.
    """
    path = Path(path)

    cfg = load_yaml(path)

    # Support inheritance
    extends = cfg.get("extends")
    merged: Dict[str, Any] = {}

    if extends:
        if isinstance(extends, (str, Path)):
            parents = [extends]
        elif isinstance(extends, list):
            parents = extends
        else:
            raise ValueError("Config key 'extends' must be a string or a list of strings.")

        for parent in parents:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = (path.parent / parent_path).resolve()
            parent_cfg = load_config(parent_path)
            merged = _deep_merge(merged, parent_cfg)

    # Merge current cfg last (wins)
    cfg_no_extends = dict(cfg)
    cfg_no_extends.pop("extends", None)
    merged = _deep_merge(merged, cfg_no_extends)

    # Add provenance
    merged.setdefault("_meta", {})
    merged["_meta"]["config_path"] = str(path.resolve())

    return merged


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """
    Creates parent directories for known output paths.
    Safe to call multiple times.

    Expected config layout:
      output:
        casas_windows: data/processed/...
        casas_meta: data/processed/...
        sphere_windows: ...
        sphere_meta: ...
    """
    output = cfg.get("output", {}) or {}
    if isinstance(output, dict):
        for _, p in output.items():
            if isinstance(p, (str, Path)) and str(p).strip():
                pp = Path(p)
                # If it's a file path, create its parent. If it's a dir path, mkdir itself.
                parent = pp if pp.suffix == "" else pp.parent
                parent.mkdir(parents=True, exist_ok=True)

    # Optional: create raw dirs if provided (won't hurt)
    datasets = cfg.get("datasets", {}) or {}
    if isinstance(datasets, dict):
        for ds in datasets.values():
            if isinstance(ds, dict) and ds.get("raw_dir"):
                Path(ds["raw_dir"]).mkdir(parents=True, exist_ok=True)
