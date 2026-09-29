"""T1.6: Lee smooths flat speckle, preserves a 1-px line where median fails; uint8 out."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from samudra.common import load_yaml
from samudra.preprocess.contrast import contrast_normalise
from samudra.preprocess.despeckle import despeckle, lee_filter

DCFG = load_yaml("configs/preprocess.yaml")["despeckle"]
CCFG = load_yaml("configs/preprocess.yaml")["contrast"]


def test_flat_speckle_std_decreases():
    rs = np.random.RandomState(0)
    flat = 100 + rs.normal(0, 15, (128, 128)).astype(np.float32)  # speckle
    out = lee_filter(flat, DCFG["win"])
    assert out.std() < flat.std(), (flat.std(), out.std())


def test_thin_line_preserved_lee_beats_median():
    bg_val, line_val = 50.0, 250.0
    img = np.full((64, 64), bg_val, np.float32)
    img[:, 32] = line_val                      # 1-px vertical bright line

    lee = lee_filter(img, DCFG["win"])
    med = median_filter(img, size=DCFG["win"])

    bg = float(np.median(lee[:, 10]))          # background away from the line
    lee_ratio = float(lee[:, 32].max()) / bg
    med_ratio = float(med[:, 32].max()) / float(np.median(med[:, 10]))

    assert lee_ratio >= 3.0, f"Lee line ratio {lee_ratio:.2f} < 3x"
    assert med_ratio < 3.0, f"median unexpectedly preserved line ({med_ratio:.2f}x)"


def test_frost_selectable():
    rs = np.random.RandomState(1)
    img = 100 + rs.normal(0, 10, (48, 48)).astype(np.float32)
    out = despeckle(img, {"method": "frost", "win": 5, "frost_damp": 2.0})
    assert out.shape == img.shape and out.std() < img.std()


def test_contrast_output_uint8_range():
    rs = np.random.RandomState(2)
    img = rs.normal(120, 40, (100, 100)).astype(np.float32)
    out = contrast_normalise(img, CCFG)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
