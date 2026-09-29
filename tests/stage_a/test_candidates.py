"""T2.2: anomaly map → candidates. Synthetic maps only — no ONNX / no dataset files
(CI-safe: the model + images are gitignored)."""

from __future__ import annotations

import numpy as np

from samudra.common import load_yaml
from samudra.stage_a.candidates import extract_candidates

CFG = load_yaml("configs/stage_a.yaml")["candidates"]


def _map_with_blobs(boxes, val=0.6, peak=0.9, bg=0.05, size=640):
    m = np.full((size, size), bg, np.float32)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = val
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2   # brighter core -> peak > mean
        m[cy - 3:cy + 3, cx - 3:cx + 3] = peak
    return m


def test_two_blobs_two_candidates():
    amap = _map_with_blobs([(100, 100, 160, 160), (400, 300, 460, 380)])
    cands = extract_candidates(amap, CFG, threshold=0.3)
    assert len(cands) == 2
    for c in cands:
        assert c.origin == "anomaly"
        assert c.anomaly_score > c.anomaly_mean             # peak strictly above mean
        assert abs(c.anomaly_score - 0.9) < 1e-3            # peak == planted core value
        assert c.polygon is not None


def test_box_matches_planted_region():
    amap = _map_with_blobs([(200, 150, 300, 250)])
    (c,) = extract_candidates(amap, CFG, threshold=0.3)
    x1, y1, x2, y2 = c.box
    assert abs(x1 - 200) <= 3 and abs(y1 - 150) <= 3
    assert abs(x2 - 300) <= 3 and abs(y2 - 250) <= 3


def test_min_area_px_drops_tiny_blob():
    # no morphology so this isolates the min-area filter
    cfg = {**CFG, "morph_kernel": 1, "min_area_px": 100}
    amap = _map_with_blobs([(10, 10, 16, 16), (300, 300, 340, 340)])  # 36px, 1600px
    cands = extract_candidates(amap, cfg, threshold=0.3)
    assert len(cands) == 1                                   # tiny one dropped
    assert (cands[0].box[2] - cands[0].box[0]) >= 30


def test_valid_mask_suppresses_blob():
    amap = _map_with_blobs([(100, 100, 200, 200)])
    valid = np.ones((640, 640), bool)
    valid[80:220, 80:220] = False                           # blob sits in invalid region
    cands = extract_candidates(amap, CFG, valid_mask=valid, threshold=0.3)
    assert cands == []                                       # suppressed, not invented


def test_min_area_m2_uses_resolution():
    # 40x40 px blob = 1600 px; at 0.1 m/px -> 0.01 m2/px -> 16 m2. min_area_m2=20 drops it.
    cfg = {**CFG, "morph_kernel": 1, "min_area_m2": 20.0}
    amap = _map_with_blobs([(300, 300, 340, 340)])
    cands = extract_candidates(amap, cfg, threshold=0.3,
                               slant_res_m_per_px=0.1, along_res_m_per_px=0.1)
    assert cands == []                                       # 16 m2 < 20 m2 min
    # a bigger blob (100x100 px = 100 m2) survives
    amap2 = _map_with_blobs([(200, 200, 300, 300)])
    cands2 = extract_candidates(amap2, cfg, threshold=0.3,
                                slant_res_m_per_px=0.1, along_res_m_per_px=0.1)
    assert len(cands2) == 1
