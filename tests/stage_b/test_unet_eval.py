"""T3.1: unet_eval pure functions — positive-class IoU/F1 + tiling offsets (CI-safe)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "unet_eval", ROOT / "scripts" / "eval" / "unet_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level imports are stdlib/cv2/numpy; onnxruntime is lazy
    return mod


def test_iou_f1_known():
    ue = _load()
    iou, f1 = ue.iou_f1(tp=10, fp=5, fn=5)
    assert abs(iou - 0.5) < 1e-6           # 10/(10+5+5)
    assert abs(f1 - (20 / 30)) < 1e-6      # 2*10/(2*10+5+5)


def test_iou_f1_perfect_and_empty():
    ue = _load()
    iou, f1 = ue.iou_f1(tp=7, fp=0, fn=0)
    assert abs(iou - 1.0) < 1e-6 and abs(f1 - 1.0) < 1e-6
    iou0, f10 = ue.iou_f1(0, 0, 0)         # no positives anywhere -> 0, no crash (R3 edge)
    assert iou0 == 0.0 and f10 == 0.0


def test_offsets_cover_edges():
    ue = _load()
    o = ue.offsets(1000, 512, 384)
    assert o[0] == 0 and o[-1] == 1000 - 512      # last window flush to the edge
    assert all(0 <= v <= 1000 - 512 for v in o)
    assert ue.offsets(400, 512, 384) == [0]       # image smaller than crop -> single window
