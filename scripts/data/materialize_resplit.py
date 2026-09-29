#!/usr/bin/env python
"""Materialise a YOLO-ready re-split tree from a manifest, via symlinks (T0.2 → training).

The upload has an images/test <-> labels/train folder-name mismatch and no test split.
This builds a clean tree with matching folder names and the new 70/15/15 split:

    _resplit/<name>/images/{train,val,test}/*.jpg   (symlinks to originals)
    _resplit/<name>/labels/{train,val,test}/*.txt   (symlinks; omitted for anomaly)
    _resplit/<name>/data.yaml                        (Ultralytics-ready)

Symlinks — no image bytes duplicated, and _resplit/ is gitignored.

    python scripts/data/materialize_resplit.py            # all three
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

CMAP = yaml.safe_load(Path("configs/class_map.yaml").read_text())
OUT = Path("_resplit")

# manifest -> (tree name, class names for data.yaml or None, has labels)
COMPONENTS = {
    "detection.csv": ("detection", CMAP["detection_names"], True),
    "segmentation.csv": ("segmentation", CMAP["segmentation_names"], True),
    "anomaly_normal.csv": ("anomaly", None, False),
}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def materialise(manifest: Path, name: str, names, has_labels: bool) -> dict:
    rows = list(csv.DictReader(manifest.open()))
    tree = OUT / name
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        split = r["split"]
        img = Path(r["image_path"])
        _link(img, tree / "images" / split / img.name)
        if has_labels and r.get("label_path"):
            lbl = Path(r["label_path"])
            if lbl.exists():
                _link(lbl, tree / "labels" / split / (img.stem + ".txt"))
        counts[split] += 1

    data_yaml = tree / "data.yaml"
    if names is not None:
        y = {
            "path": str(tree.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": names,
        }
        header = ""
        if name == "segmentation":
            header = (
                "# CAVEAT (R4): SYNTHETIC-ONLY. shipwreck class ships 0 real masks — only\n"
                "# synthetic ghost_net present. Real UNet IoU (T3.1) stays AWAITING a real\n"
                "# AI4Shipwrecks mask download. Do NOT report seg IoU as a real number.\n"
            )
        data_yaml.write_text(header + yaml.safe_dump(y, sort_keys=False))
    else:
        # anomaly: no labels; AE trains on images/train (object-free), evals on test
        data_yaml.write_text(
            "# AnomalyScreen normal_seabed — object-free SYNTHETIC seabed for the T2.1\n"
            "# autoencoder. CAVEAT: sim-to-real gap may inflate false alarms; prefer real\n"
            "# AI4Shipwrecks extras/terrain when available. 'object' positives for the\n"
            "# AUROC eval come from the detection test split, not here.\n"
            f"train: {(tree / 'images/train').resolve()}\n"
            f"val: {(tree / 'images/val').resolve()}\n"
            f"test: {(tree / 'images/test').resolve()}\n"
        )
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    a = ap.parse_args()
    for fname, (name, names, has_labels) in COMPONENTS.items():
        man = a.manifests / fname
        if not man.exists():
            print(f"[skip] {man} missing — run index_samudra_v2.py first")
            continue
        counts = materialise(man, name, names, has_labels)
        print(f"[ok] {name:12s} -> {OUT / name}  {counts}")
    print("\nUltralytics: yolo train data=_resplit/detection/data.yaml model=yolo11n.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
