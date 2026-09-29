"""Speckle reduction that preserves thin net lines (T1.6).

Adaptive Lee (default) keeps detail where local variance is high (edges, 1-px net
lines) and smooths flat speckle where it is low — unlike a median filter, which erases
thin lines. Frost is selectable via config.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def lee_filter(img: np.ndarray, win: int) -> np.ndarray:
    x = img.astype(np.float32)
    mean = uniform_filter(x, size=win, mode="nearest")
    mean_sq = uniform_filter(x * x, size=win, mode="nearest")
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    noise_var = float(local_var.mean())  # dominated by flat regions
    k = np.clip((local_var - noise_var) / (local_var + 1e-6), 0.0, 1.0)
    return mean + k * (x - mean)


def frost_filter(img: np.ndarray, win: int, damp: float) -> np.ndarray:
    """Simplified Frost: exponential spatial weights whose decay grows with local
    coefficient-of-variation (more variance -> less smoothing)."""
    x = img.astype(np.float32)
    mean = uniform_filter(x, size=win, mode="nearest")
    mean_sq = uniform_filter(x * x, size=win, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci = var / (mean * mean + 1e-6)          # squared coeff of variation per pixel
    r = win // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(yy * yy + xx * xx)
    out = np.zeros_like(x)
    wsum = np.zeros_like(x)
    h, w = x.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
            weight = np.exp(-damp * ci * dist[dy + r, dx + r])
            out += weight * shifted
            wsum += weight
    return out / (wsum + 1e-6)


def despeckle(img: np.ndarray, cfg: dict) -> np.ndarray:
    method = cfg.get("method", "lee")
    win = int(cfg.get("win", 5))
    if method == "lee":
        return lee_filter(img, win)
    if method == "frost":
        return frost_filter(img, win, float(cfg.get("frost_damp", 2.0)))
    raise ValueError(f"unknown despeckle method: {method}")
