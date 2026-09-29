"""Overlapping tiling with exact coordinate mapping back to the full image (T1.8).

Each Tile stores its (row_off, col_off) and a cropped valid_mask, so tile-local
coordinates map back to full-image coordinates with zero error.
"""

from __future__ import annotations

import numpy as np

from samudra.common.types import Tile


def _offsets(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    offs = list(range(0, length - size + 1, stride))
    if offs[-1] != length - size:
        offs.append(length - size)  # final tile flush to the edge (still ~overlap)
    return offs


def tile_image(
    pixels: np.ndarray,
    cfg: dict,
    valid_mask: np.ndarray | None = None,
) -> list[Tile]:
    h, w = pixels.shape
    size = int(cfg.get("size", 640))
    overlap = float(cfg.get("overlap", 0.20))
    stride = max(1, int(round(size * (1 - overlap))))
    min_valid = float(cfg.get("min_valid_frac", 0.0))
    if valid_mask is None:
        valid_mask = np.ones((h, w), bool)

    tiles: list[Tile] = []
    for r in _offsets(h, size, stride):
        for c in _offsets(w, size, stride):
            r2, c2 = min(r + size, h), min(c + size, w)
            vm = valid_mask[r:r2, c:c2]
            if vm.mean() < min_valid:
                continue
            tiles.append(
                Tile(
                    tile_id=f"r{r}_c{c}",
                    pixels=pixels[r:r2, c:c2].copy(),
                    row_off=r,
                    col_off=c,
                    valid_mask=vm.copy(),
                )
            )
    return tiles
