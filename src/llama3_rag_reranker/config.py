"""Configuration loading (YAML + CLI overrides) and seeding.

Heavy libraries are imported lazily inside ``set_seed`` so this module stays
import-light for CI.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper over a nested dict with dotted-key access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @property
    def data(self) -> dict[str, Any]:
        return self._data


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI override strings."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply ``key=value`` CLI overrides (dotted keys)."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        cfg.set(key.strip(), _coerce(value.strip()))
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) Torch/MLX for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto") -> str:
    """Resolve an embedding/model device. 'auto' prefers MPS, then CPU."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def project_root() -> Path:
    """Repository root (two levels up from this file's package)."""
    return Path(__file__).resolve().parents[2]
