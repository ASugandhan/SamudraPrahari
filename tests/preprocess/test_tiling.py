"""T1.8: tiling round-trips coordinates with 0 px error; overlap + valid_mask stored."""

from __future__ import annotations

import numpy as np

from samudra.common import load_yaml
from samudra.preprocess.tiling import tile_image

CFG = load_yaml("configs/preprocess.yaml")["tiling"]


def test_coordinate_roundtrip_zero_error():
    rs = np.random.RandomState(0)
    img = rs.randint(0, 255, (1000, 800), np.uint8)
    tiles = tile_image(img, CFG)
    assert len(tiles) > 1
    for t in tiles:
        th, tw = t.pixels.shape
        # every tile pixel maps back to the identical full-image pixel (0 error)
        for (lr, lc) in [(0, 0), (th - 1, tw - 1), (th // 2, tw // 3)]:
            fr, fc = t.to_full(lr, lc)
            assert (fr, fc) == (lr + t.row_off, lc + t.col_off)
            assert img[int(fr), int(fc)] == t.pixels[lr, lc]


def test_overlap_present():
    img = np.zeros((2000, 2000), np.uint8)
    tiles = tile_image(img, CFG)
    row_offs = sorted({t.row_off for t in tiles})
    size = CFG["size"]
    stride = int(round(size * (1 - CFG["overlap"])))
    diffs = [b - a for a, b in zip(row_offs, row_offs[1:], strict=False)]
    # tiles overlap: every consecutive offset is < one tile apart
    assert all(d < size for d in diffs)
    # regular interior spacing equals the configured stride
    assert stride in diffs and stride < size


def test_valid_mask_stored_and_min_frac_skips():
    img = np.ones((1300, 700), np.uint8)
    valid = np.ones((1300, 700), bool)
    valid[:640, :] = False              # top band invalid -> first row of tiles empty
    tiles = tile_image(img, {**CFG, "min_valid_frac": 0.5}, valid)
    assert all(t.valid_mask is not None for t in tiles)
    # no fully-invalid tile survived
    assert all(t.valid_mask.mean() >= 0.5 for t in tiles)


def test_small_image_single_tile():
    img = np.zeros((100, 120), np.uint8)
    tiles = tile_image(img, CFG)
    assert len(tiles) == 1
    assert tiles[0].pixels.shape == (100, 120)
