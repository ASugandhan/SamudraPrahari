#!/usr/bin/env python
"""Dataset card from data/manifests/*.csv (T0.2).

Reports counts per split, and — per Rule R4 — REAL vs SYNTHETIC vs UNVERIFIED
separately, including per-class counts per split (so real-class test counts, the
numbers headline metrics are quoted from, are explicit).

    python scripts/data/dataset_card.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAN = Path("data/manifests")
SPLITS = ["train", "val", "test"]


def _class_split_table(rows, origin=None):
    tab = defaultdict(Counter)  # class -> split -> file count
    for r in rows:
        if origin and r.get("origin") != origin:
            continue
        classes = [c for c in (r.get("classes") or "").split("|") if c]
        for c in classes:
            tab[c][r["split"]] += 1
    return tab


def _print_table(title, tab):
    if not tab:
        return
    print(f"    {title}")
    print(f"      {'class':12s} " + "".join(f"{s:>7s}" for s in SPLITS) + "   total")
    for c in sorted(tab):
        counts = [tab[c].get(s, 0) for s in SPLITS]
        print(f"      {c:12s} " + "".join(f"{n:7d}" for n in counts) + f"   {sum(counts)}")


def main() -> int:
    csvs = sorted(p for p in MAN.glob("*.csv") if not p.name.startswith("_"))
    if not csvs:
        print(f"no manifests in {MAN}/ — run index_samudra_v2.py (see reports/T0.2.md)")
        return 0
    for c in csvs:
        rows = list(csv.DictReader(c.open()))
        n_groups = len({r["source_group_id"] for r in rows})
        per_split = Counter(r["split"] for r in rows)
        per_origin = Counter(r.get("origin", "?") for r in rows)
        print(f"\n=== {c.stem} ===  images={len(rows)}  groups={n_groups}")
        print(f"  split : {{{', '.join(f'{s}:{per_split.get(s,0)}' for s in SPLITS)}}}")
        print(f"  origin: {dict(per_origin)}   (R4: report real & synthetic separately)")
        # per-class breakdown by origin
        _print_table("REAL classes (headline numbers use these):",
                     _class_split_table(rows, "real"))
        _print_table("UNVERIFIED (do not cite as real):",
                     _class_split_table(rows, "unverified"))
        _print_table("SYNTHETIC (report separately):",
                     _class_split_table(rows, "synthetic"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
