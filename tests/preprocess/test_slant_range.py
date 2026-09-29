"""T1.3: slant→ground correction places targets at their true ground range (±1 px),
using each ping's own altitude."""

from __future__ import annotations

import numpy as np

from samudra.common import load_yaml
from samudra.preprocess.slant_range import slant_to_ground

CFG = load_yaml("configs/preprocess.yaml")["slant_range"]


def _slant_row_with_target(w: int, c0: int, alt: float, ground_target: float) -> np.ndarray:
    """A slant-range row: bright target at slant = sqrt(ground² + alt²) on starboard."""
    row = np.full(w, 20.0, np.float32)
    s = int(round(np.sqrt(ground_target**2 + alt**2)))
    col = c0 + s
    if 0 <= col < w:
        row[col] = 255
    return row


def test_target_lands_at_ground_range():
    w, c0, alt, g = 400, 200, 30.0, 80.0
    img = np.tile(_slant_row_with_target(w, c0, alt, g), (10, 1))
    out, valid = slant_to_ground(img, np.full(10, alt, np.float32), CFG)
    # brightest output column on starboard side should be ~ c0 + ground_target
    star = out[5, c0:]
    peak = int(np.argmax(star))
    assert abs(peak - g) <= 1, f"target at ground {peak}, expected {g}"


def test_uses_per_ping_altitude():
    # two pings, different altitudes, same ground target -> both land at same ground col
    w, c0, g = 400, 200, 80.0
    alts = np.array([20.0, 50.0], np.float32)
    rows = np.stack([_slant_row_with_target(w, c0, a, g) for a in alts])
    out, _ = slant_to_ground(rows, alts, CFG)
    peaks = [int(np.argmax(out[i, c0:])) for i in range(2)]
    assert all(abs(p - g) <= 1 for p in peaks), f"per-ping altitude not used: {peaks}"


def test_water_column_masked_not_invented():
    # a large altitude makes far-range slants exceed the swath -> those cols masked
    w, alt = 200, 40.0
    img = np.full((4, w), 100.0, np.float32)
    out, valid = slant_to_ground(img, np.full(4, alt, np.float32), CFG)
    # some output columns must be invalid (masked), and masked pixels == fill_value
    assert (~valid).any()
    assert np.all(out[~valid] == CFG["fill_value"])
