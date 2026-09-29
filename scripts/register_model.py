#!/usr/bin/env python
"""Validate an ONNX model and register it (T0.3).

Checks the file loads in onnxruntime on CPU (Rule R11), computes its sha256, and
writes/updates models/registry.json. Rejects missing/corrupt ONNX with a clear error.

    python scripts/register_model.py --onnx models/stage_a_ae.onnx --name stage_a_ae \
        --version v1 --trained-on "AI4Shipwrecks terrain" --metrics models/stage_a_metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REGISTRY = Path("models/registry.json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def validate_onnx_cpu(path: Path) -> tuple[list[str], list[str]]:
    """Load in onnxruntime on CPU. Returns (input_names, output_names). Raises on failure."""
    import onnxruntime as ort  # lazy: package usable without onnxruntime installed

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()]


def register(
    onnx: Path,
    name: str,
    version: str,
    trained_on: str | None,
    metrics: Path | None,
    registry: Path = REGISTRY,
    now: str | None = None,
) -> dict:
    if not onnx.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx}")
    try:
        inputs, outputs = validate_onnx_cpu(onnx)
    except Exception as e:  # noqa: BLE001 — surface any ORT load failure clearly
        raise ValueError(f"ONNX failed to load on CPU ({onnx}): {e}") from e

    entry = {
        "version": version,
        "onnx_path": str(onnx),
        "sha256": sha256_file(onnx),
        "trained_on": trained_on,
        "metrics_file": str(metrics) if metrics else None,
        "date": now or dt.date.today().isoformat(),
        "inputs": inputs,
        "outputs": outputs,
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    reg = json.loads(registry.read_text()) if registry.exists() else {}
    reg[name] = entry
    registry.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--trained-on", default=None)
    ap.add_argument("--metrics", default=None, type=Path)
    a = ap.parse_args()
    try:
        entry = register(a.onnx, a.name, a.version, a.trained_on, a.metrics)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[ok] registered {a.name}: sha256={entry['sha256'][:12]} "
          f"inputs={entry['inputs']} outputs={entry['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
