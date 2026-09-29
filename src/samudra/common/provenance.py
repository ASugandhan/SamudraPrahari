"""Provenance stamping (Rule R7): every output records which model/version/config
produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import config_hash


@dataclass
class Provenance:
    model_name: str
    model_version: str
    config_hash: str
    extra: dict | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        if d.get("extra") is None:
            d.pop("extra")
        return d


def stamp(model_name: str, model_version: str, config, **extra) -> dict:
    """Build a provenance dict from a model id and the config that drove it."""
    return Provenance(
        model_name=model_name,
        model_version=model_version,
        config_hash=config_hash(config) if config is not None else "none",
        extra=extra or None,
    ).as_dict()
