"""T1.8: canonical pipeline runs steps in the fixed order and is the only public
entry point (Rule R8)."""

from __future__ import annotations

import inspect

import numpy as np

from samudra.common.types import SonarFrame
from samudra.preprocess import pipeline

CANONICAL = ["bottom", "slant", "nadir", "gain", "despeckle", "contrast", "quality", "tiling"]


def _frame(h=300, w=400):
    rs = np.random.RandomState(0)
    px = rs.randint(0, 255, (h, w), np.uint8)
    return SonarFrame(pixels=px, mode="full", source_file="synthetic",
                      slant_res_m_per_px=0.05)


def test_step_order_constant_matches_document():
    assert pipeline.STEP_ORDER == CANONICAL


def test_pipeline_runs_in_order():
    res = pipeline.run(_frame())
    assert res.steps_run == CANONICAL
    assert len(res.tiles) >= 1
    # every tile carries a quality flag (quality gate ran before tiling emitted them)
    assert all(t.quality_flag in ("ok", "low") for t in res.tiles)


def test_only_one_public_pipeline_function():
    public_funcs = [
        name for name, obj in inspect.getmembers(pipeline, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == pipeline.__name__
    ]
    assert public_funcs == ["run"], f"expected only run(), found {public_funcs}"


def test_steps_are_toggleable_order_preserved():
    res = pipeline.run(_frame(), enabled={"slant": False, "gain": False})
    assert res.steps_run == [s for s in CANONICAL if s not in ("slant", "gain")]
