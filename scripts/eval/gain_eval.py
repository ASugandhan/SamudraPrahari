#!/usr/bin/env python
"""Gain/EGN + destripe evaluation (T1.5): CV of column-mean profile before/after,
stripe-energy before/after, and before/after PNGs.

With --file: runs on a real image/sonar file. Without: a synthetic survey with a known
across-track fall-off and an injected stripe (evidence until a real survey is fetched).

    python scripts/eval/gain_eval.py
    python scripts/eval/gain_eval.py --file data/raw/.../survey.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from samudra.common import load_yaml
from samudra.preprocess.gain import bac_egn, column_cv, fft_destripe, normalise_gain

CFG = load_yaml("configs/preprocess.yaml")["gain"]
OUT = Path("out")


def _synthetic():
    rs = np.random.RandomState(0)
    h, w = 300, 400
    x = np.linspace(-1, 1, w)
    falloff = 40 + 200 * np.exp(-(x**2) / 0.2)
    img = (falloff[None, :] * rs.normal(1.0, 0.05, (h, w))).astype(np.float32)
    stripe = 30 * np.sin(2 * np.pi * 25 * np.arange(w) / w)
    return img + stripe[None, :]


def _save(name, arr):
    OUT.mkdir(exist_ok=True)
    a = np.clip(arr, 0, None)
    a = (255 * (a - a.min()) / (np.ptp(a) + 1e-6)).astype(np.uint8)
    Image.fromarray(a).save(OUT / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    if a.file:
        img = np.asarray(Image.open(a.file).convert("L"), np.float32)
        tag = Path(a.file).stem
    else:
        img, tag = _synthetic(), "synthetic"

    cv0 = column_cv(img)
    after_gain = bac_egn(img, CFG)
    cv1 = column_cv(after_gain)
    final = fft_destripe(after_gain, CFG)

    def axis_stripe_energy(im):
        f = np.fft.fftshift(np.fft.fft2(im))
        cy = im.shape[0] // 2
        row = np.abs(f[cy])
        cx = im.shape[1] // 2
        band = 3
        row[cx - band : cx + band + 1] = 0
        return float(row.max())

    print(f"[{tag}] column-mean CV: {cv0:.4f} -> {cv1:.4f} "
          f"({100 * (1 - cv1 / cv0):.0f}% reduction; target ≥50%)")
    print(f"[{tag}] peak across-track stripe energy: "
          f"{axis_stripe_energy(img):.0f} -> {axis_stripe_energy(final):.0f}")
    _save(f"{tag}_before.png", img)
    _save(f"{tag}_after.png", normalise_gain(img, CFG))
    print(f"saved out/{tag}_before.png, out/{tag}_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
