"""Contrast normalisation: percentile clip + mild CLAHE → uint8 0–255 (T1.6)."""

from __future__ import annotations

import cv2
import numpy as np


def contrast_normalise(img: np.ndarray, cfg: dict) -> np.ndarray:
    x = img.astype(np.float32)
    lo = np.percentile(x, cfg.get("clip_lo_pct", 1))
    hi = np.percentile(x, cfg.get("clip_hi_pct", 99))
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    u8 = (x * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.get("clahe_clip", 2.0)),
        tileGridSize=(int(cfg.get("clahe_grid", 8)), int(cfg.get("clahe_grid", 8))),
    )
    return clahe.apply(u8)
