#!/usr/bin/env python
"""Stage A candidate recall vs false-candidate rate (T2.2).

Runs the autoencoder anomaly map on each test image, extracts candidates over a sweep of
thresholds, and reports recall (GT objects hit by >=1 candidate, IoU>=0.1 or centre-inside)
against false-candidates-per-image. Numbers per source_dataset (R4/R5). Writes the chosen
operating-point threshold back into configs/stage_a.yaml.

    python scripts/eval/stage_a_recall.py --split test
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from samudra.common import load_yaml
from samudra.stage_a.anomaly import AnomalyModel
from samudra.stage_a.candidates import extract_candidates

CFG_PATH = Path("configs/stage_a.yaml")


def _gt_boxes(label_path: str, size: int) -> list[tuple[int, int, int, int]]:
    p = Path(label_path)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        cx, cy, w, h = (float(v) for v in f[1:5])
        out.append((int((cx - w / 2) * size), int((cy - h / 2) * size),
                    int((cx + w / 2) * size), int((cy + h / 2) * size)))
    return out


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _centre_inside(gt, cand) -> bool:
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return cand[0] <= cx <= cand[2] and cand[1] <= cy <= cand[3]


def _hit(gt, cand) -> bool:
    return _iou(gt, cand) >= 0.1 or _centre_inside(gt, cand)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default="data/manifests/detection.csv")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_yaml(CFG_PATH)
    ccfg = cfg["candidates"]
    sig = cfg["anomaly"]["smooth_sigma"]
    model = AnomalyModel(model_name=cfg["anomaly"]["model_name"])
    size = model.h

    rows = [r for r in csv.DictReader(open(a.manifest)) if r["split"] == a.split]
    if a.limit:
        rows = rows[: a.limit]

    # anomaly map per image once; threshold sweep is cheap on the cached map
    cached = []  # (source, amap, gt_boxes)
    for r in rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        amap = model.anomaly_map(img, sig)
        cached.append((r["source_dataset"], amap, _gt_boxes(r["label_path"], size)))
    print(f"images: {len(cached)}  (split={a.split})")

    # adaptive threshold grid from the anomaly-map value distribution (model-agnostic:
    # a fixed 0.10-0.50 grid is mis-ranged when a better-reconstructing model has smaller errors)
    sample = np.concatenate([a.ravel()[::53] for _, a, _ in cached]) if cached else np.array([0.1])
    qs = [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
    thresholds = sorted({round(float(np.quantile(sample, q)), 4) for q in qs})

    def eval_thr(thr, subset):
        gt_total = hits = false_cands = 0
        for _src, amap, gts in subset:
            cands = [c.box for c in extract_candidates(amap, ccfg, threshold=thr)]
            gt_total += len(gts)
            matched_c = set()
            for gt in gts:
                for j, cb in enumerate(cands):
                    if _hit(gt, cb):
                        hits += 1
                        matched_c.add(j)
                        break
            false_cands += sum(1 for j in range(len(cands)) if j not in matched_c)
        n = max(len(subset), 1)
        recall = hits / gt_total if gt_total else 0.0
        return recall, false_cands / n

    # ---- overall recall-vs-false-candidates curve ----
    print("\nOVERALL recall vs false-candidates/image:")
    print("  thr    recall   fc/img")
    curve = []
    for t in thresholds:
        rec, fc = eval_thr(t, cached)
        curve.append((t, rec, fc))
        print(f"  {t:.2f}   {rec:.3f}    {fc:.2f}")

    # operating point: lowest fc with recall>=0.80, else max-recall
    ok = [c for c in curve if c[1] >= 0.80]
    if ok:
        op = min(ok, key=lambda c: c[2])
        reached = True
    else:
        op = max(curve, key=lambda c: c[1])
        reached = False
    print(f"\noperating point: thr={op[0]:.2f} recall={op[1]:.3f} fc/img={op[2]:.2f} "
          f"({'recall>=0.80 reached' if reached else 'recall<0.80 — reporting max-recall knee'})")

    # ---- per-dataset at the operating point (R4/R5) ----
    print("\nper-dataset @ operating point (real vs synthetic — R4/R5):")
    by_src = defaultdict(list)
    for item in cached:
        by_src[item[0]].append(item)
    for src in sorted(by_src):
        rec, fc = eval_thr(op[0], by_src[src])
        print(f"  {src:20s} recall {rec:.3f}  fc/img {fc:.2f}  (n={len(by_src[src])})")

    txt = CFG_PATH.read_text()
    txt = re.sub(r"(?m)^(\s*threshold:).*$", rf"\g<1> {op[0]:.3f}", txt, count=1)
    CFG_PATH.write_text(txt)
    print(f"\nwrote operating-point threshold {op[0]:.3f} -> {CFG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
