"""T2.3: yolo_eval R4 real-vs-synthetic split logic (CI-safe — no ultralytics/weights)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "yolo_eval", ROOT / "scripts" / "eval" / "yolo_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level imports are stdlib only; ultralytics is lazy in main()
    return mod


def test_class_origin_map():
    ye = _load()
    assert ye.CLASS_ORIGIN["shipwreck"] == "real"
    assert ye.CLASS_ORIGIN["aircraft"] == "real"
    assert ye.CLASS_ORIGIN["ghost_net"] == "synth"
    assert ye.REAL_CLASSES == {"shipwreck", "aircraft"}


def test_real_synth_map50_split():
    ye = _load()
    rows = [
        {"class": "shipwreck", "AP50": 0.80},
        {"class": "aircraft", "AP50": 0.60},   # REAL mean = 0.70
        {"class": "ghost_net", "AP50": 0.90},
        {"class": "cylinder", "AP50": 0.50},
        {"class": "pipe", "AP50": 0.40},        # SYNTH mean = 0.60
    ]
    real_m, synth_m = ye.real_synth_map50(rows)
    assert abs(real_m - 0.70) < 1e-9
    assert abs(synth_m - 0.60) < 1e-9


def test_real_synth_map50_handles_empty():
    ye = _load()
    real_m, synth_m = ye.real_synth_map50([{"class": "ghost_net", "AP50": 0.9}])
    assert real_m is None and abs(synth_m - 0.9) < 1e-9
