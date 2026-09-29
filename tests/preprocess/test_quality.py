"""T1.7: quality score separates good vs poor tiles; balanced accuracy ≥ 0.75."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from samudra.common import load_yaml
from samudra.preprocess.quality import quality_flag, quality_score

ROOT = Path(__file__).resolve().parents[2]
CFG = load_yaml("configs/quality.yaml")


def _load_eval():
    spec = importlib.util.spec_from_file_location(
        "quality_eval", ROOT / "scripts" / "eval" / "quality_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_in_unit_range():
    qe = _load_eval()
    for t in (qe._good(0), qe._poor(1000)):
        s, _ = quality_score(t, CFG)
        assert 0.0 <= s <= 1.0


def test_good_scores_higher_than_poor():
    qe = _load_eval()
    good = quality_score(qe._good(3), CFG)[0]
    poor = quality_score(qe._poor(1002), CFG)[0]
    assert good > poor


def test_balanced_accuracy_threshold():
    qe = _load_eval()
    tiles, labels = qe.synthetic_set()
    scores = [quality_score(t, CFG)[0] for t in tiles]
    thr, ba = qe.calibrate(scores, labels)
    assert ba >= 0.75, f"balanced accuracy {ba:.2f} < 0.75"
    # flag helper agrees with the calibrated threshold direction
    assert quality_flag(thr - 0.01, {"threshold": thr}) == "low"
    assert quality_flag(thr + 0.01, {"threshold": thr}) == "ok"
