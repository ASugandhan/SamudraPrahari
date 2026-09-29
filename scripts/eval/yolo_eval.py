#!/usr/bin/env python
"""YOLO detector evaluation on the test split (T2.3).

Per-class AP/P/R + mAP50/mAP50-95, with REAL vs SYNTHETIC reported separately (R4), and an
optional per-source breakdown (R5) that gives the SCTD-specific mAP50 the master prompt asks for.

Needs `ultralytics` + trained weights (run on Colab, or locally after `pip install ultralytics`
and copying best.pt back). The ONNX is for deployment; metrics come from the .pt via val().

    python scripts/eval/yolo_eval.py --split test --weights runs/detect/train/weights/best.pt
    python scripts/eval/yolo_eval.py --split test --weights best.pt --per-source
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path

# canonical class -> provenance for the R4-honest split (headline = REAL only)
CLASS_ORIGIN = {
    "shipwreck": "real", "aircraft": "real",
    "pipe": "synth+unver", "cylinder": "synth", "ghost_net": "synth",
}
REAL_CLASSES = {"shipwreck", "aircraft"}


def per_class_rows(res) -> list[dict]:
    """Extract per-class metrics from an Ultralytics val() result."""
    box = res.box
    rows = []
    for i, c in enumerate(box.ap_class_index):
        name = res.names[int(c)]
        rows.append({
            "class": name, "origin": CLASS_ORIGIN.get(name, "?"),
            "AP50": float(box.ap50[i]), "AP50_95": float(box.ap[i]),
            "P": float(box.p[i]), "R": float(box.r[i]),
        })
    return rows


def real_synth_map50(rows: list[dict]) -> tuple[float | None, float | None]:
    """Mean AP50 over REAL classes and over the rest (R4 headline split)."""
    real = [r["AP50"] for r in rows if r["class"] in REAL_CLASSES]
    synth = [r["AP50"] for r in rows if r["class"] not in REAL_CLASSES]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return _mean(real), _mean(synth)


def _print_table(rows: list[dict]) -> None:
    print(f"  {'class':12s} {'origin':11s} {'AP50':>7s} {'AP50-95':>8s} {'P':>6s} {'R':>6s}")
    for r in sorted(rows, key=lambda x: x["class"]):
        print(f"  {r['class']:12s} {r['origin']:11s} {r['AP50']:7.3f} {r['AP50_95']:8.3f} "
              f"{r['P']:6.3f} {r['R']:6.3f}")


def _subset_yaml(rows, names, tmp: Path) -> Path:
    """Build a temp YOLO tree (symlinks) + data.yaml for a subset of test rows."""
    import yaml
    for sub in ("images/test", "labels/test"):
        (tmp / sub).mkdir(parents=True, exist_ok=True)
    for r in rows:
        img, lbl = Path(r["image_path"]), Path(r["label_path"])
        (tmp / "images/test" / img.name).symlink_to(img.resolve())
        if lbl.exists():
            (tmp / "labels/test" / (img.stem + ".txt")).symlink_to(lbl.resolve())
    y = {"path": str(tmp.resolve()), "train": "images/test", "val": "images/test",
         "test": "images/test", "nc": len(names), "names": names}
    p = tmp / "data.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--data", default="_resplit/detection/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default="data/manifests/detection.csv")
    ap.add_argument("--per-source", action="store_true")
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. `pip install ultralytics` (or run on Colab).")
        return 1
    if not Path(a.weights).exists():
        print(f"ERROR: weights not found: {a.weights} (train first / copy best.pt back).")
        return 1

    import yaml
    names = yaml.safe_load(Path(a.data).read_text())["names"]
    model = YOLO(a.weights)

    print(f"=== YOLO {Path(a.weights).name} — {a.split} split (overall) ===")
    res = model.val(data=a.data, split=a.split, imgsz=640, verbose=False)
    rows = per_class_rows(res)
    _print_table(rows)
    real_m, synth_m = real_synth_map50(rows)
    print(f"  overall mAP50 {res.box.map50:.3f}  mAP50-95 {res.box.map:.3f}")
    print(f"  R4 headline -> REAL mAP50 {real_m if real_m is None else round(real_m,3)}  |  "
          f"SYNTH mAP50 {synth_m if synth_m is None else round(synth_m,3)}")

    if a.per_source:
        by_src = defaultdict(list)
        for r in csv.DictReader(open(a.manifest)):
            if r["split"] == a.split:
                by_src[r["source_dataset"]].append(r)
        print("\n=== per-source (R5); gives SCTD test mAP50 for EXPECTED #1 ===")
        print(f"  {'source':20s} {'n':>4s} {'mAP50':>7s} {'mAP50-95':>9s}")
        for src in sorted(by_src):
            with tempfile.TemporaryDirectory() as td:
                yml = _subset_yaml(by_src[src], names, Path(td))
                r = model.val(data=str(yml), split="test", imgsz=640, verbose=False)
                print(f"  {src:20s} {len(by_src[src]):4d} {r.box.map50:7.3f} {r.box.map:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
