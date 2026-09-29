"""T1.2: bottom tracking recovers per-ping altitude, no NaN, spikes suppressed."""

from __future__ import annotations

import numpy as np

from samudra.common import load_yaml
from samudra.preprocess.bottom_tracking import track_bottom_array

CFG = load_yaml("configs/preprocess.yaml")["bottom_tracking"]


def make_waterfall(alt_per_ping: np.ndarray, w: int = 200, rng_seed: int = 0) -> np.ndarray:
    """Synthetic port|starboard waterfall: water column then bright first return."""
    rs = np.random.RandomState(rng_seed)
    h = alt_per_ping.size
    c0 = w // 2
    img = np.full((h, w), 8.0, np.float32)  # dark water column baseline
    for r in range(h):
        a = int(round(alt_per_ping[r]))
        for side in (-1, +1):
            for d in range(w // 2):
                col = c0 + side * d
                if not (0 <= col < w):
                    continue
                if d < a:
                    img[r, col] = 8
                elif d < a + 3:
                    img[r, col] = 220        # bright first return band
                else:
                    img[r, col] = 90         # seabed
    img += rs.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


def test_altitude_mae_within_10pct():
    # varying altitude across pings (ramp) — exercises per-ping tracking
    alt = np.linspace(28, 40, 120).astype(np.float32)
    img = make_waterfall(alt)
    tracked = track_bottom_array(img, CFG)
    mae = float(np.mean(np.abs(tracked - alt)))
    assert mae <= 0.10 * alt.mean(), f"MAE {mae:.2f}px > 10% of {alt.mean():.1f}"


def test_no_nan_output():
    alt = np.full(60, 30.0, np.float32)
    tracked = track_bottom_array(make_waterfall(alt), CFG)
    assert not np.any(np.isnan(tracked))


def test_spikes_suppressed():
    # inject a spike ping; median filter + clamp must bound the jump
    alt = np.full(80, 30.0, np.float32)
    alt[40] = 90.0  # outlier
    tracked = track_bottom_array(make_waterfall(alt), CFG)
    jumps = np.abs(np.diff(tracked))
    assert jumps.max() <= CFG["max_jump_px"] + 1e-6


def test_tracks_varying_altitude():
    alt = np.linspace(25, 45, 100).astype(np.float32)
    tracked = track_bottom_array(make_waterfall(alt), CFG)
    # monotone-ish: strong positive correlation with the true ramp
    assert np.corrcoef(tracked, alt)[0, 1] > 0.95
