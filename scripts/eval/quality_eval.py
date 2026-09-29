#!/usr/bin/env python
"""Calibrate + evaluate the quality gate (T1.7).

If data/quality_labels.csv exists (from scripts/label_quality.py on 60 real tiles),
uses those. Otherwise runs on a synthetic good/poor set (evidence until humans label).
Prints balanced accuracy + confusion matrix, and writes the calibrated threshold back
into configs/quality.yaml.

    python scripts/eval/quality_eval.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

from samudra.common import load_yaml
from samudra.preprocess.quality import quality_score

CFG_PATH = Path("configs/quality.yaml")
LABELS = Path("data/quality_labels.csv")


# ---- synthetic tiles (good vs poor) ----

def _good(seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:128, 0:128]
    base = 60 + 120 * (xx / 128.0)                       # seabed gradient
    blob = 90 * np.exp(-((yy - 64) ** 2 + (xx - 70) ** 2) / (2 * 12**2))  # a feature
    img = base + blob + rs.normal(0, 4, (128, 128))       # low noise -> high SNR
    return np.clip(img, 0, 255).astype(np.uint8)


def _poor(seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    kind = seed % 3
    if kind == 0:                                         # washed-out low contrast
        img = 120 + rs.normal(0, 2, (128, 128))
    elif kind == 1:                                       # speckle-dominated (low SNR)
        img = 128 + rs.normal(0, 55, (128, 128))
    else:                                                 # heavy dropout bands
        img = 60 + 120 * (np.mgrid[0:128, 0:128][1] / 128.0)
        img[::2, :] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_set(n: int = 30):
    tiles = [_good(i) for i in range(n)] + [_poor(1000 + i) for i in range(n)]
    labels = [1] * n + [0] * n                            # 1 = good, 0 = poor
    return tiles, labels


# ---- metrics ----

def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tpr = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0
    tnr = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else 0.0
    return float((tpr + tnr) / 2)


def confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def calibrate(scores, labels) -> tuple[float, float]:
    """Return (threshold, balanced_accuracy) maximising balanced accuracy."""
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(min(scores), max(scores), 101):
        pred = [1 if s >= t else 0 for s in scores]
        ba = balanced_accuracy(labels, pred)
        if ba > best_ba:
            best_t, best_ba = float(t), ba
    return best_t, best_ba


def _set_threshold_in_yaml(path: Path, value: float) -> None:
    txt = path.read_text()
    txt = re.sub(r"(?m)^threshold:.*$", f"threshold: {value:.3f}", txt, count=1)
    path.write_text(txt)


def main() -> int:
    cfg = load_yaml(CFG_PATH)
    if LABELS.exists():
        rows = list(csv.DictReader(LABELS.open()))
        tiles = [np.asarray(Image.open(r["tile_path"]).convert("L")) for r in rows]
        labels = [1 if r["label"] == "good" else 0 for r in rows]
        source = f"{LABELS} ({len(rows)} human-labelled tiles)"
    else:
        tiles, labels = synthetic_set()
        source = "SYNTHETIC good/poor set (no human labels yet — T1.7 AWAITING real 60)"

    scores = [quality_score(t, cfg)[0] for t in tiles]
    thr, ba = calibrate(scores, labels)
    pred = [1 if s >= thr else 0 for s in scores]
    cm = confusion(labels, pred)

    print(f"source            : {source}")
    print(f"calibrated thresh : {thr:.3f}")
    print(f"balanced accuracy : {ba:.3f}   (target ≥ 0.75)")
    print(f"confusion matrix  : {cm}  (1=good,0=poor)")
    _set_threshold_in_yaml(CFG_PATH, thr)
    print(f"wrote threshold -> {CFG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
