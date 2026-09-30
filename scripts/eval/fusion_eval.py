#!/usr/bin/env python
"""Candidate fusion evaluation (T2.4).

Runs YOLO ∪ AE on the test split and reports recall of YOLO-alone, AE-alone, and FUSED
(GT hit by ≥1 candidate; IoU≥0.1 or centre-inside), plus the duplicate rate (target ≤5%).
Fused recall must be ≥ max(YOLO, AE) — fusion is a union. All coords in 640² space so YOLO
(scaled), AE (native 640), and GT (normalised×640) are comparable.

    python scripts/eval/fusion_eval.py --split test \
        --weights out/yolo11n_best.pt --data _resplit/detection/data.yaml
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import yaml

from samudra.common import load_yaml
from samudra.common.types import Candidate
from samudra.stage_a.anomaly import AnomalyModel
from samudra.stage_a.candidates import extract_candidates
from samudra.stage_a.fusion import box_iou, fuse_candidates

SIZE = 640


def gt_boxes(label_path: str, size: int) -> list[tuple[int, int, int, int]]:
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


def hits(gt, cand_box) -> bool:
    if box_iou(gt, cand_box) >= 0.1:
        return True
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return cand_box[0] <= cx <= cand_box[2] and cand_box[1] <= cy <= cand_box[3]


def _recall(gts, cand_boxes) -> int:
    return sum(1 for g in gts if any(hits(g, b) for b in cand_boxes))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default="out/yolo11n_best.pt")
    ap.add_argument("--data", default="_resplit/detection/data.yaml")
    ap.add_argument("--manifest", default="data/manifests/detection.csv")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed (pip install ultralytics or run on Colab).")
        return 1
    if not Path(a.weights).exists():
        print(f"ERROR: weights not found: {a.weights}")
        return 1

    cfg = load_yaml("configs/stage_a.yaml")
    ccfg, fcfg, sig = cfg["candidates"], cfg["fusion"], cfg["anomaly"]["smooth_sigma"]
    names = yaml.safe_load(Path(a.data).read_text())["names"]
    ae = AnomalyModel(model_name=cfg["anomaly"]["model_name"])
    yolo = YOLO(a.weights)

    rows = [r for r in csv.DictReader(open(a.manifest)) if r["split"] == a.split]
    if a.limit:
        rows = rows[: a.limit]

    gt_total = h_yolo = h_ae = h_fused = n_fused = dup = 0
    origins = Counter()
    for r in rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        sx, sy = SIZE / W, SIZE / H

        ae_c = extract_candidates(ae.anomaly_map(img, sig), ccfg)  # 640 space
        res = yolo(r["image_path"], verbose=False)[0]
        yolo_c = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            yolo_c.append(Candidate(
                box=(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
                tile_id=Path(r["image_path"]).stem, origin="yolo",
                yolo_class=names[int(b.cls)], yolo_conf=float(b.conf)))

        fused = fuse_candidates(yolo_c, ae_c, fcfg)
        origins.update(c.origin for c in fused)

        gts = gt_boxes(r["label_path"], SIZE)
        gt_total += len(gts)
        h_yolo += _recall(gts, [c.box for c in yolo_c])
        h_ae += _recall(gts, [c.box for c in ae_c])
        per_gt = [sum(1 for c in fused if hits(g, c.box)) for g in gts]
        h_fused += sum(1 for m in per_gt if m > 0)
        dup += sum(max(0, m - 1) for m in per_gt)
        n_fused += len(fused)

    r_yolo = h_yolo / gt_total if gt_total else 0.0
    r_ae = h_ae / gt_total if gt_total else 0.0
    r_fused = h_fused / gt_total if gt_total else 0.0
    dup_rate = dup / n_fused if n_fused else 0.0

    print(f"images {len(rows)}  GT objects {gt_total}")
    print(f"recall  YOLO-alone {r_yolo:.3f} | AE-alone {r_ae:.3f} | FUSED {r_fused:.3f}")
    print(f"fused >= max(yolo, ae)? {r_fused >= max(r_yolo, r_ae) - 1e-9}")
    print(f"duplicate rate {dup_rate:.3f} (target <= 0.05)  [dup {dup} / {n_fused} fused]")
    print(f"fused origins {dict(origins)}  (anomaly_only = UNKNOWN pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
