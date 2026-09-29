"""No-reference tile quality score from interpretable proxies (T1.7).

Proxies: estimated SNR (low-freq signal vs high-freq residual), contrast (p95−p5),
edge/contour density, dropout fraction (all-zero rows/cols), saturation fraction.
Combined by config weights into quality_score ∈ [0,1]. Low-scoring tiles are flagged
"low" (re-survey) — they still run downstream but carry a warning + severity cap.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def _snr(x: np.ndarray) -> float:
    signal = uniform_filter(x, size=5, mode="nearest")
    noise = x - signal
    ns = noise.std()
    return float(signal.std() / ns) if ns > 1e-6 else 10.0


def _dropout_fraction(x: np.ndarray) -> float:
    row_zero = np.all(x <= 1, axis=1).mean()
    col_zero = np.all(x <= 1, axis=0).mean()
    return float(max(row_zero, col_zero))


def quality_proxies(tile: np.ndarray) -> dict:
    x = tile.astype(np.float32)
    p5, p95 = np.percentile(x, 5), np.percentile(x, 95)
    edges = cv2.Canny(tile.astype(np.uint8), 50, 150)
    sat = ((x <= 2) | (x >= 253)).mean()
    return {
        "snr": _snr(x),
        "contrast": float((p95 - p5) / 255.0),
        "edge_density": float((edges > 0).mean()),
        "dropout": _dropout_fraction(x),
        "saturation": float(sat),
    }


def quality_score(tile: np.ndarray, cfg: dict) -> tuple[float, dict]:
    p = quality_proxies(tile)
    w = cfg["weights"]
    snr_n = p["snr"] / (p["snr"] + cfg.get("snr_half", 1.0))
    edge_n = min(1.0, p["edge_density"] / cfg.get("edge_ref", 0.15))
    components = {
        "snr": w["snr"] * snr_n,
        "contrast": w["contrast"] * min(1.0, p["contrast"]),
        "edge_density": w["edge_density"] * edge_n,
        "dropout": w["dropout"] * (1.0 - p["dropout"]),
        "saturation": w["saturation"] * (1.0 - p["saturation"]),
    }
    score = float(sum(components.values()))
    return score, {"proxies": p, "components": components}


def quality_flag(score: float, cfg: dict) -> str:
    return "low" if score < cfg.get("threshold", 0.5) else "ok"
