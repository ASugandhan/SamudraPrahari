"""T1.4: nadir mask width follows per-ping altitude; boolean valid_mask propagated."""

from __future__ import annotations

import numpy as np

from samudra.common import load_yaml
from samudra.preprocess.nadir import mask_nadir

CFG = load_yaml("configs/preprocess.yaml")["nadir"]


def test_mask_width_follows_altitude():
    w = 300
    alts = np.array([10.0, 30.0, 60.0], np.float32)
    img = np.full((3, w), 100.0, np.float32)
    out, valid = mask_nadir(img, alts, CFG)
    pad = CFG["pad_px"]
    for r, a in enumerate(alts):
        masked_cols = int((~valid[r]).sum())
        # gap half-width = a + pad on each side -> ~2*(a+pad) columns
        assert abs(masked_cols - 2 * (a + pad)) <= 2, (r, masked_cols, a)
    # wider altitude -> wider mask
    widths = [int((~valid[r]).sum()) for r in range(3)]
    assert widths[0] < widths[1] < widths[2]


def test_masked_pixels_zeroed_not_inpainted():
    img = np.full((2, 100), 200.0, np.float32)
    out, valid = mask_nadir(img, np.array([15.0, 15.0], np.float32), CFG)
    assert np.all(out[~valid] == 0)          # zeroed
    assert np.all(out[valid] == 200.0)       # untouched elsewhere (not inpainted)


def test_valid_mask_combined_with_incoming():
    img = np.ones((1, 100), np.float32)
    incoming = np.ones((1, 100), bool)
    incoming[0, :5] = False                  # already-invalid region survives
    out, valid = mask_nadir(img, np.array([10.0], np.float32), CFG, valid_mask=incoming)
    assert not valid[0, :5].any()
