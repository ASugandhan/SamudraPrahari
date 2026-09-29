"""Nadir / water-column masking after slant correction (T1.4).

The near-nadir ground region (ground range < altitude) is specular/unreliable. We
mask it per ping — width follows that ping's altitude — setting pixels to 0 and
recording a boolean valid_mask. Never inpainted (R1/R2).
"""

from __future__ import annotations

import numpy as np


def mask_nadir(
    pixels: np.ndarray,
    altitude_px: np.ndarray,
    cfg: dict,
    nadir_col: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (masked_pixels, valid_mask). Nadir gap half-width per ping = altitude + pad."""
    h, w = pixels.shape
    c0 = w // 2 if nadir_col is None else int(nadir_col)
    pad = int(cfg.get("pad_px", 0))

    ground = np.abs(np.arange(w) - c0)[None, :]          # (1, w) ground range per col
    half_width = (altitude_px[:, None] + pad)            # (h, 1) per ping
    nadir_gap = ground < half_width                      # True = inside gap -> mask

    valid = np.ones((h, w), bool) if valid_mask is None else valid_mask.copy()
    valid &= ~nadir_gap

    out = pixels.copy()
    out[~valid] = 0
    return out, valid
