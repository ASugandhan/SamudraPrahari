"""Gain normalisation (BAC/EGN) + FFT stripe removal (T1.5).

Implemented from the published method description (BAC = mean intensity per incidence
angle; EGN = mean per angle × range bin), NOT copied from GPL `sidescantools`.

For a single ground-range survey image the across-track position IS the range/incidence
bin, so BAC and EGN both reduce to a per-column gain table (mean intensity per column over
all pings); dividing by it flattens the across-track brightness fall-off. Along-track
stripe noise shows up as bright lines through the origin of the 2-D FFT; we notch those
axes outside a small DC band.
"""

from __future__ import annotations

import numpy as np


def bac_egn(pixels: np.ndarray, cfg: dict, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Divide out the across-track (per-range-bin) mean gain profile."""
    x = pixels.astype(np.float32)
    mask = np.ones_like(x, bool) if valid_mask is None else valid_mask
    counts = np.maximum(mask.sum(axis=0), 1)
    col_mean = (x * mask).sum(axis=0) / counts  # per-column (per-range-bin) mean
    valid_cols = col_mean > cfg.get("eps", 1e-6)
    global_mean = col_mean[valid_cols].mean() if valid_cols.any() else 1.0
    gain = np.where(valid_cols, col_mean / global_mean, 1.0)
    gain = np.where(gain < cfg.get("eps", 1e-6), 1.0, gain)
    return x / gain[None, :]


def column_cv(pixels: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """Coefficient of variation of the across-track column-mean profile (flatness metric)."""
    x = pixels.astype(np.float32)
    mask = np.ones_like(x, bool) if valid_mask is None else valid_mask
    counts = np.maximum(mask.sum(axis=0), 1)
    col_mean = (x * mask).sum(axis=0) / counts
    col_mean = col_mean[col_mean > 0]
    if col_mean.size == 0 or col_mean.mean() == 0:
        return 0.0
    return float(col_mean.std() / col_mean.mean())


def fft_destripe(pixels: np.ndarray, cfg: dict) -> np.ndarray:
    """Notch periodic stripe energy on the two central FFT axes (outside a DC band).

    ponytail: whole-axis notch (classic SSS destripe); it also removes any true signal
    that is perfectly constant along one axis — acceptable for seabed. Narrow-band
    peak-only notch is the upgrade path if it ever clips real structure.
    """
    h, w = pixels.shape
    f = np.fft.fftshift(np.fft.fft2(pixels.astype(np.float32)))
    cy, cx = h // 2, w // 2
    bw_x = max(1, int(cfg.get("fft_notch_frac", 0.02) * w))
    bw_y = max(1, int(cfg.get("fft_notch_frac", 0.02) * h))
    # fy=0 row (across-track periodic stripes) outside DC band
    f[cy, : cx - bw_x] = 0
    f[cy, cx + bw_x + 1 :] = 0
    # fx=0 column (along-track periodic banding) outside DC band
    f[: cy - bw_y, cx] = 0
    f[cy + bw_y + 1 :, cx] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(f))).astype(np.float32)


def normalise_gain(pixels: np.ndarray, cfg: dict, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """BAC/EGN then FFT destripe — the T1.5 step used by the pipeline."""
    return fft_destripe(bac_egn(pixels, cfg, valid_mask), cfg)
