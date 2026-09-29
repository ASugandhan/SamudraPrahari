#!/usr/bin/env python
"""Quick CLI to label tiles good/poor for quality-gate calibration (T1.7, human step).

Shows each tile's interpretable proxies and asks good/poor. Writes data/quality_labels.csv
(columns: tile_path,label). Label ~60 tiles, then run scripts/eval/quality_eval.py.

    python scripts/label_quality.py --tiles out/tiles           # interactive
    python scripts/label_quality.py --tiles out/tiles --open     # also open each image
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from samudra.common import load_yaml
from samudra.preprocess.quality import quality_proxies

OUT = Path("data/quality_labels.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, type=Path)
    ap.add_argument("--open", action="store_true", help="open each image while labelling")
    a = ap.parse_args()
    load_yaml("configs/quality.yaml")  # sanity: config present

    tiles = sorted(p for p in a.tiles.rglob("*.png"))
    if not tiles:
        print(f"no tiles in {a.tiles}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {r["tile_path"] for r in csv.DictReader(OUT.open())}

    with OUT.open("a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["tile_path", "label"])
        for p in tiles:
            if str(p) in done:
                continue
            img = np.asarray(Image.open(p).convert("L"))
            pr = quality_proxies(img)
            print(f"\n{p}")
            print("  " + "  ".join(f"{k}={v:.3f}" for k, v in pr.items()))
            if a.open:
                Image.open(p).show()
            ans = input("  good/poor/skip [g/p/s]? ").strip().lower()
            if ans.startswith("g"):
                w.writerow([str(p), "good"])
            elif ans.startswith("p"):
                w.writerow([str(p), "poor"])
            f.flush()
    print(f"\nlabels -> {OUT}. Now run scripts/eval/quality_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
