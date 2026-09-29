#!/usr/bin/env python
"""Index the uploaded Samudra_Prahari_Dataset_v2 into leak-free manifests (T0.2).

The upload is already YOLO-format and pre-split (train in a misnamed images/test folder,
labels in labels/train). We RE-SPLIT 70/15/15 by group (user decision), grouping per
source (Rule R5): ai4 by site, sctd/synthetic per image, pipe20xx as one group. Each
source is split independently so every class lands in every split. Real vs synthetic is
tagged per row (Rule R4). The provider's original split is preserved in `provider_split`.

Writes: data/manifests/{detection,segmentation,anomaly_normal}.csv

    python scripts/data/index_samudra_v2.py --root Samudra_Prahari_Dataset_v2
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import yaml

from samudra.common import sha256_file
from samudra.common.splits import split_by_group

COLUMNS = [
    "image_path", "label_path", "source_dataset", "source_group_id",
    "split", "provider_split", "label_type", "origin", "classes", "sha256",
]

CMAP = yaml.safe_load(Path("configs/class_map.yaml").read_text())
SOURCES = CMAP["sources"]
DET_NAMES = CMAP["detection_names"]
SEG_NAMES = CMAP["segmentation_names"]


def _source_of(stem: str) -> str:
    for pre in ("pure_synth_subpipe", "pure_synth_uatd", "ai4", "sctd", "pipe20xx",
                "synth_net", "normal"):
        if stem.startswith(pre):
            return "normal_seabed" if pre == "normal" else pre
    return "other"


def _group_id(source: str, stem: str) -> str:
    if source == "ai4":
        return re.sub(r"_\d+$", "", stem)          # ai4_DM_Wilson_03 -> ai4_DM_Wilson (site)
    if source == "pipe20xx":
        return "pipe20xx_2010"                      # single unverified survey -> one group
    return stem                                     # sctd / synthetic / normal: per image


def _classes_from_label(label_path: Path, names: list[str]) -> str:
    if not label_path or not label_path.exists():
        return ""
    ids = sorted({int(line.split()[0]) for line in label_path.read_text().splitlines()
                  if line.strip()})
    return "|".join(names[i] for i in ids if 0 <= i < len(names))


def _rows_detection(root: Path) -> list[dict]:
    det = root / "Detection"
    # provider layout: images/test <-> labels/train (train), images/val <-> labels/val
    layout = [("test", "train", "train"), ("val", "val", "val")]
    rows = []
    for img_sub, lbl_sub, provider_split in layout:
        for img in sorted((det / "images" / img_sub).glob("*.jpg")):
            lbl = det / "labels" / lbl_sub / (img.stem + ".txt")
            src = _source_of(img.stem)
            rows.append({
                "image_path": str(img), "label_path": str(lbl),
                "source_dataset": src, "source_group_id": _group_id(src, img.stem),
                "provider_split": provider_split, "label_type": "box",
                "origin": SOURCES.get(src, {}).get("origin", "unknown"),
                "classes": _classes_from_label(lbl, DET_NAMES),
            })
    return rows


def _rows_segmentation(root: Path) -> list[dict]:
    seg = root / "Segmentation"
    rows = []
    for split in ("train", "val"):
        for img in sorted((seg / "images" / split).glob("*.jpg")):
            lbl = seg / "labels" / split / (img.stem + ".txt")
            src = _source_of(img.stem)
            rows.append({
                "image_path": str(img), "label_path": str(lbl),
                "source_dataset": src, "source_group_id": _group_id(src, img.stem),
                "provider_split": split, "label_type": "mask",
                "origin": SOURCES.get(src, {}).get("origin", "unknown"),
                "classes": _classes_from_label(lbl, SEG_NAMES),
            })
    return rows


def _rows_anomaly(root: Path) -> list[dict]:
    ans = root / "AnomalyScreen" / "normal_seabed"
    rows = []
    for img in sorted(ans.glob("*.jpg")):
        rows.append({
            "image_path": str(img), "label_path": "",
            "source_dataset": "normal_seabed", "source_group_id": img.stem,
            "provider_split": "unsplit", "label_type": "none",
            "origin": "synthetic", "classes": "",
        })
    return rows


def _assign_splits(rows: list[dict], seed: int) -> None:
    """Per-source grouped 70/15/15 so every source (class) appears in every split."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source_dataset"]].append(r)
    for src, srows in by_source.items():
        assign = split_by_group([r["source_group_id"] for r in srows], seed=seed)
        for r in srows:
            r["split"] = assign[r["source_group_id"]]


def _finalize(rows: list[dict], seed: int, out: Path, do_sha: bool) -> None:
    _assign_splits(rows, seed)
    for r in rows:
        r["sha256"] = sha256_file(Path(r["image_path"])) if do_sha else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print(f"[ok] {len(rows):5d} rows -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("Samudra_Prahari_Dataset_v2"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-sha", action="store_true", help="skip sha256 (faster, for dev)")
    a = ap.parse_args()
    do_sha = not a.no_sha
    _finalize(_rows_detection(a.root), a.seed, Path("data/manifests/detection.csv"), do_sha)
    _finalize(_rows_segmentation(a.root), a.seed, Path("data/manifests/segmentation.csv"), do_sha)
    _finalize(_rows_anomaly(a.root), a.seed, Path("data/manifests/anomaly_normal.csv"), do_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
