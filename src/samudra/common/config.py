"""YAML config loader with optional pydantic schema validation (Rule R9).

All thresholds live in configs/*.yaml, never as code constants. Use `load_config`
with a pydantic model to validate, or `load_yaml` for a plain dict.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config {p} must be a mapping, got {type(data).__name__}")
    return data


def load_config(path: str | Path, model: type[T]) -> T:
    """Load YAML and validate against a pydantic model (raises on schema error)."""
    return model.model_validate(load_yaml(path))


def config_hash(cfg: dict | BaseModel) -> str:
    """Stable short hash of a config, for provenance stamping (Rule R7)."""
    if isinstance(cfg, BaseModel):
        cfg = cfg.model_dump(mode="json")
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
