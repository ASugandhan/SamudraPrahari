"""Model registry tests (T0.3): valid ONNX registers, missing/corrupt is rejected."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]


def _load_register_model():
    spec = importlib.util.spec_from_file_location(
        "register_model", ROOT / "scripts" / "register_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_dummy_onnx(path: Path) -> None:
    """Minimal identity graph: input x -> output y (loads on CPU)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "dummy", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # match onnxruntime's supported IR range (R11 CPU load)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_register_valid(tmp_path):
    rm = _load_register_model()
    onnx_path = tmp_path / "dummy.onnx"
    make_dummy_onnx(onnx_path)
    reg = tmp_path / "registry.json"
    entry = rm.register(onnx_path, "dummy", "v1", "unit-test", None, registry=reg, now="2026-01-01")
    assert entry["inputs"] == ["x"] and entry["outputs"] == ["y"]
    assert len(entry["sha256"]) == 64
    saved = json.loads(reg.read_text())
    assert saved["dummy"]["version"] == "v1"


def test_register_missing_file(tmp_path):
    rm = _load_register_model()
    with pytest.raises(FileNotFoundError):
        rm.register(tmp_path / "nope.onnx", "x", "v1", None, None, registry=tmp_path / "r.json")


def test_register_corrupt_file(tmp_path):
    rm = _load_register_model()
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    with pytest.raises(ValueError, match="failed to load"):
        rm.register(bad, "x", "v1", None, None, registry=tmp_path / "r.json")


def test_onnxruntime_cpu_inference(tmp_path):
    """R11: exported model must run on CPU via onnxruntime."""
    import onnxruntime as ort

    rm = _load_register_model()  # noqa: F841 — ensures module import path works
    onnx_path = tmp_path / "dummy.onnx"
    make_dummy_onnx(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": np.ones((1, 3), np.float32)})[0]
    assert np.allclose(out, 1.0)
