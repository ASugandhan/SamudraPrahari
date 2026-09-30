#!/usr/bin/env python
"""UNet segmentation eval — IoU + F1 of the POSITIVE class only (T3.1, Rule R3).

Runs the exported UNet ONNX (CPU, no torch/smp needed) over the tiled AI4Shipwrecks test
split and aggregates positive-class TP/FP/FN → IoU, F1. NEVER reports pixel accuracy
(99.9% of pixels are seabed — R3). Preprocessing matches training exactly: grayscale →
3-channel → /255 → ImageNet mean/std.

    python scripts/eval/unet_eval.py --split test --real-only
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def offsets(n: int, c: int, s: int) -> list[int]:
    if n <= c:
        return [0]
    o = list(range(0, n - c + 1, s))
    if o[-1] != n - c:
        o.append(n - c)
    return o


def iou_f1(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Positive-class IoU and F1 (R3). eps guards the all-empty case."""
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return float(iou), float(f1)


def _preprocess(crop_gray: np.ndarray) -> np.ndarray:
    x = np.repeat(crop_gray[..., None], 3, 2).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # [1,3,H,W]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/stage_b_unet.onnx")
    ap.add_argument("--ai4", default="AI4Shipwrecks")
    ap.add_argument("--split", default="test")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--real-only", action="store_true", help="eval on real AI4 masks (default)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if not Path(a.onnx).exists():
        print(f"ERROR: {a.onnx} not found — train T3.1 on Colab and copy it back first.")
        return 1
    import onnxruntime as ort

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    c = a.crop

    imgs = sorted(glob.glob(f"{a.ai4}/{a.split}/images/*.png"))
    if a.limit:
        imgs = imgs[: a.limit]
    if not imgs:
        print(f"ERROR: no test images under {a.ai4}/{a.split}/images")
        return 1

    tp = fp = fn = 0
    for ip in imgs:
        stem = Path(ip).name
        mp = f"{a.ai4}/{a.split}/labels/{stem}"
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or m is None:
            continue
        H, W = im.shape
        for y in offsets(H, c, a.stride):
            for x in offsets(W, c, a.stride):
                ci = im[y:y + c, x:x + c]
                cm = m[y:y + c, x:x + c] > 0
                if ci.shape != (c, c):
                    continue
                logits = sess.run([oname], {iname: _preprocess(ci)})[0][0, 0]
                pred = (1.0 / (1.0 + np.exp(-logits))) > 0.5
                tp += int(np.logical_and(pred, cm).sum())
                fp += int(np.logical_and(pred, ~cm).sum())
                fn += int(np.logical_and(~pred, cm).sum())

    iou, f1 = iou_f1(tp, fp, fn)
    print(f"AI4Shipwrecks {a.split} (REAL): IoU(+) {iou:.3f}  F1(+) {f1:.3f}   "
          f"(benchmark SOTA ~0.445 / UNet ~0.411; target >=0.40)")
    print(f"  positive-class only (R3 — NOT pixel accuracy).  images={len(imgs)}  "
          f"TP={tp} FP={fp} FN={fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
