"""Slant-range → ground-range correction, per-ping altitude (T1.3).

For a ping at altitude a, a seabed point at ground range g lies at slant range
s = sqrt(g² + a²). We resample each row so output columns are uniform in ground
range: output ground g ← input slant sqrt(g² + a²) (backward map for cv2.remap).
Output columns whose required slant falls outside the swath are masked, never
invented (the water-column region, slant < altitude, has no ground equivalent).
"""

from __future__ import annotations

import cv2
import numpy as np


def slant_to_ground(
    pixels: np.ndarray,
    altitude_px: np.ndarray,
    cfg: dict,
    nadir_col: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ground_img, valid_mask). altitude_px is per-ping (len == n rows)."""
    h, w = pixels.shape
    c0 = w // 2 if nadir_col is None else int(nadir_col)
    assert altitude_px.shape[0] == h, "altitude must be per-ping"

    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), bool)

    cols = np.arange(w)
    ground = np.abs(cols - c0).astype(np.float32)  # ground range per output col
    side = np.sign(cols - c0)
    for r in range(h):
        a = float(altitude_px[r])
        slant = np.sqrt(ground * ground + a * a)      # required input slant distance
        in_col = c0 + side * slant
        map_x[r] = in_col
        map_y[r] = r
        valid[r] = (in_col >= 0) & (in_col <= w - 1)

    out = cv2.remap(
        pixels.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(cfg.get("fill_value", 0)),
    )
    out[~valid] = cfg.get("fill_value", 0)
    return out.astype(pixels.dtype), valid
