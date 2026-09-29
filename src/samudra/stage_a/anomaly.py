"""Stage A anomaly map: run the autoencoder, take smoothed per-pixel reconstruction
error (T2.1 inference side, consumed by T2.2 candidate extraction).

The AE (models/stage_a_ae.onnx) reconstructs object-free seabed. Where the input
differs from its reconstruction, error is high — that is the anomaly signal. Input is
[0,1] grayscale at the model's native size (640x640, sigmoid output), matching training.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def _resolve_onnx(model_name: str, registry: str | Path) -> str:
    reg = json.loads(Path(registry).read_text())
    if model_name not in reg:
        raise KeyError(f"{model_name} not in {registry}")
    return reg[model_name]["onnx_path"]


class AnomalyModel:
    """Thin ONNX-Runtime (CPU) wrapper producing anomaly maps in the model's native space."""

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        model_name: str = "stage_a_ae",
        registry: str | Path = "models/registry.json",
    ):
        import onnxruntime as ort  # lazy: package importable without onnxruntime

        if onnx_path is None:
            onnx_path = _resolve_onnx(model_name, registry)
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.iname = inp.name
        self.oname = self.sess.get_outputs()[0].name
        # [1, 1, H, W]
        self.h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 640

    def _preprocess(self, tile: np.ndarray) -> np.ndarray:
        gray = tile if tile.ndim == 2 else cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        r = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return (r.astype(np.float32) / 255.0)[None, None]  # [1,1,H,W] in [0,1]

    def reconstruct(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._preprocess(tile)
        recon = self.sess.run([self.oname], {self.iname: x})[0][0, 0]
        return x[0, 0], recon  # (input, reconstruction), both [H,W] in [0,1]

    def anomaly_map(self, tile: np.ndarray, smooth_sigma: float = 4.0) -> np.ndarray:
        """Smoothed per-pixel L1 reconstruction error, in the model's native HxW space."""
        inp, recon = self.reconstruct(tile)
        err = np.abs(inp - recon)
        if smooth_sigma and smooth_sigma > 0:
            err = gaussian_filter(err, smooth_sigma)
        return err.astype(np.float32)
