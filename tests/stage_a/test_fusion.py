"""T2.4: candidate fusion (YOLO ∪ AE) — synthetic candidates, CI-safe (no model/data)."""

from __future__ import annotations

from samudra.common.types import Candidate
from samudra.stage_a.fusion import box_iou, fuse_candidates

CFG = {"merge_iou": 0.5}


def _yolo(box, cls="shipwreck", conf=0.9):
    return Candidate(box=box, tile_id="t", origin="yolo", yolo_class=cls, yolo_conf=conf)


def _ae(box, score=0.7):
    return Candidate(box=box, tile_id="t", origin="anomaly", anomaly_score=score, anomaly_mean=0.4)


def test_overlapping_merge_to_both_keeps_both_scores():
    y = _yolo((100, 100, 200, 200))
    a = _ae((105, 105, 205, 205))                 # IoU high with y
    fused = fuse_candidates([y], [a], CFG)
    assert len(fused) == 1
    c = fused[0]
    assert c.origin == "both"
    assert c.yolo_class == "shipwreck" and c.yolo_conf == 0.9   # YOLO score kept
    assert c.anomaly_score == 0.7                                # AE score kept


def test_yolo_only_and_ae_only_tags():
    y = _yolo((0, 0, 50, 50))
    a = _ae((400, 400, 460, 460))                 # no overlap
    fused = fuse_candidates([y], [a], CFG)
    origins = sorted(c.origin for c in fused)
    assert origins == ["anomaly_only", "yolo"]     # AE-only -> UNKNOWN pool tag


def test_union_completeness():
    ys = [_yolo((0, 0, 40, 40)), _yolo((300, 300, 360, 360))]
    aes = [_ae((2, 2, 42, 42)), _ae((500, 500, 560, 560))]   # 1st overlaps ys[0] (IoU~0.82), 2nd alone
    fused = fuse_candidates(ys, aes, CFG)
    # 2 yolo (one becomes 'both') + 1 ae-only = 3 total
    assert len(fused) == 3
    assert sum(c.origin == "both" for c in fused) == 1
    assert sum(c.origin == "yolo" for c in fused) == 1
    assert sum(c.origin == "anomaly_only" for c in fused) == 1


def test_each_ae_merges_at_most_once():
    # one AE candidate overlapping two YOLO boxes must merge into only one of them
    y1 = _yolo((100, 100, 200, 200))
    y2 = _yolo((150, 150, 250, 250))
    a = _ae((140, 140, 240, 240))
    fused = fuse_candidates([y1, y2], [a], CFG)
    assert len(fused) == 2                          # no phantom extra candidate
    assert sum(c.origin == "both" for c in fused) == 1
    assert sum(c.origin == "yolo" for c in fused) == 1


def test_box_iou_basic():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
