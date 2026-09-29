"""Bottom tracking → per-ping altitude (T1.2).

Side-scan pings have a low-intensity water column near nadir, then a strong first
return at the seabed. Altitude (in slant pixels) is the nadir→first-return distance.
We detect it per ping on each side of nadir, then median-filter across pings and clamp
spikes. Altitude is always measured, never a fixed constant (Rule R2).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from samudra.common.logging import get_logger
from samudra.common.types import SonarFrame

log = get_logger(__name__)


def _first_return_offset(half: np.ndarray, cfg: dict) -> float:
    """Distance (px) from nadir (index 0) to the first strong seabed return.

    `half` runs outward from nadir. Returns np.nan if no clear return is found.
    """
    if half.size < 2:
        return np.nan
    sm = uniform_filter1d(half.astype(np.float32), size=cfg["smooth_win"], mode="nearest")
    rng = sm.max() - sm.min()
    if rng <= 0:
        return np.nan
    level = sm.min() + cfg["rise_frac"] * rng
    grad = np.gradient(sm)
    min_alt = cfg["min_altitude_px"]
    for i in range(min_alt, sm.size):
        if sm[i] >= level and grad[i] >= cfg["min_grad_frac"] * rng:
            return float(i)
    return np.nan


def track_bottom_array(pixels: np.ndarray, cfg: dict, nadir_col: int | None = None) -> np.ndarray:
    """Per-ping altitude in slant pixels for a port|starboard image (nadir near centre)."""
    h, w = pixels.shape
    c0 = w // 2 if nadir_col is None else int(nadir_col)
    alt = np.full(h, np.nan, np.float32)
    for r in range(h):
        row = pixels[r]
        port = row[:c0][::-1]        # outward from nadir, port side
        stbd = row[c0:]              # outward from nadir, starboard side
        offs = [_first_return_offset(port, cfg), _first_return_offset(stbd, cfg)]
        offs = [o for o in offs if not np.isnan(o)]
        if offs:
            alt[r] = float(np.mean(offs))

    # fill any gaps by nearest-valid before smoothing (never leave NaN — EXPECTED #2)
    alt = _fill_nan(alt)
    # median filter across pings to kill spikes, then clamp ping-to-ping jumps
    alt = median_filter(alt, size=cfg["median_win"], mode="nearest")
    alt = _clamp_jumps(alt, cfg["max_jump_px"])
    return alt.astype(np.float32)


def _fill_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    idx = np.arange(a.size)
    good = ~np.isnan(a)
    if not good.any():
        return np.zeros_like(a)  # no returns anywhere -> 0 altitude, flagged upstream
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def _clamp_jumps(a: np.ndarray, max_jump: float) -> np.ndarray:
    out = a.copy()
    for i in range(1, out.size):
        d = out[i] - out[i - 1]
        if abs(d) > max_jump:
            out[i] = out[i - 1] + np.sign(d) * max_jump
    return out


def track_bottom(frame: SonarFrame, cfg: dict) -> SonarFrame:
    """Populate `altitude_px` (and `altitude_m` if resolution known) on the frame."""
    alt_px = track_bottom_array(frame.pixels, cfg)
    frame.altitude_px = alt_px

    if frame.altitude_m is not None and not np.all(np.isnan(frame.altitude_m)):
        # file already had altitude — compute both, log the difference (T1.2 DO)
        recorded = frame.altitude_m
        if frame.slant_res_m_per_px:
            tracked_m = alt_px * frame.slant_res_m_per_px
            mae = float(np.nanmean(np.abs(tracked_m - recorded)))
            log.info("bottom tracking vs recorded altitude: MAE=%.3f m", mae)
    else:
        if frame.slant_res_m_per_px:
            frame.altitude_m = alt_px * frame.slant_res_m_per_px
        frame.altitude_source = "image_estimate" if frame.mode == "degraded" else "bottom_tracking"
    return frame
