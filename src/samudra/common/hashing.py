"""Shared file hashing (manifests + model registry)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()
