"""T1.5: BAC/EGN flattens across-track profile (CV -50%+); FFT notch cuts stripe energy."""

from __future__ import annotations

import numpy as np

from samudra.common import load_yaml
from samudra.preprocess.gain import bac_egn, column_cv, fft_destripe

CFG = load_yaml("configs/preprocess.yaml")["gain"]


def test_bac_egn_reduces_cv_by_half():
    rs = np.random.RandomState(0)
    h, w = 200, 256
    # across-track brightness fall-off (bright centre, dark edges) + texture
    x = np.linspace(-1, 1, w)
    falloff = 40 + 200 * np.exp(-(x**2) / 0.2)      # strong systematic profile
    texture = rs.normal(1.0, 0.05, (h, w))
    img = (falloff[None, :] * texture).astype(np.float32)

    cv_before = column_cv(img)
    out = bac_egn(img, CFG)
    cv_after = column_cv(out)
    assert cv_after <= 0.5 * cv_before, f"CV {cv_before:.3f} -> {cv_after:.3f} (need ≥50% cut)"


def test_fft_destripe_cuts_stripe_energy():
    rs = np.random.RandomState(1)
    h, w = 256, 256
    base = rs.normal(100, 5, (h, w)).astype(np.float32)
    # across-track periodic stripe: varies over columns, constant over rows -> fy=0 axis
    fx0 = 20
    cols = np.arange(w)
    stripe = 30 * np.sin(2 * np.pi * fx0 * cols / w)
    striped = base + stripe[None, :]

    def stripe_energy(im):
        f = np.fft.fftshift(np.fft.fft2(im))
        cy, cx = h // 2, w // 2
        return abs(f[cy, cx + fx0]) + abs(f[cy, cx - fx0])

    e_before = stripe_energy(striped)
    cleaned = fft_destripe(striped, CFG)
    e_after = stripe_energy(cleaned)
    assert e_after < 0.05 * e_before, f"stripe energy {e_before:.0f} -> {e_after:.0f}"


def test_destripe_preserves_dc_scale():
    rs = np.random.RandomState(2)
    img = rs.normal(100, 3, (128, 128)).astype(np.float32)
    out = fft_destripe(img, CFG)
    # overall brightness roughly preserved (DC band kept)
    assert abs(out.mean() - img.mean()) < 1.0
