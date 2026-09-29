"""Manifest tests (T0.2): R5 no-leakage guarantee + sha256 integrity.

Covers split_by_group and the Samudra-v2 indexer's per-source grouped split (always),
and validates any real manifests present in data/manifests/ (leakage always; sha256
integrity only for files present locally — images are gitignored, so CI stays green).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from samudra.common.splits import split_by_group

ROOT = Path(__file__).resolve().parents[1]


def _load_indexer():
    spec = importlib.util.spec_from_file_location(
        "index_samudra_v2", ROOT / "scripts" / "data" / "index_samudra_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- R5: split_by_group never leaks a group across splits ----

def test_split_no_group_leakage():
    groups = [f"g{i}" for i in range(200)]
    assign = split_by_group(groups, seed=1337)
    # each group -> exactly one split
    assert set(assign) == set(groups)
    splits = Counter(assign.values())
    # roughly 70/15/15
    assert splits["train"] > splits["val"]
    assert splits["train"] > splits["test"]
    assert set(splits) == {"train", "val", "test"}


def test_split_deterministic():
    g = [f"s{i}" for i in range(50)]
    assert split_by_group(g, seed=7) == split_by_group(g, seed=7)
    assert split_by_group(g, seed=7) != split_by_group(g, seed=8)


# ---- Samudra-v2 indexer: grouping rules + per-source grouped split (R5) ----

def test_indexer_group_id_rules():
    idx = _load_indexer()
    assert idx._group_id("ai4", "ai4_DM_Wilson_03") == "ai4_DM_Wilson"   # site, not frame
    assert idx._group_id("ai4", "ai4_WP_Rend_11") == "ai4_WP_Rend"
    assert idx._group_id("pipe20xx", "pipe20xx_0256_2010") == "pipe20xx_2010"  # one group
    assert idx._group_id("sctd", "sctd_000006") == "sctd_000006"          # per image
    assert idx._group_id("synth_net", "synth_net_0000") == "synth_net_0000"


def test_indexer_per_source_grouped_split():
    idx = _load_indexer()
    rows = []
    # ai4: 12 sites x 5 frames (frames of a site share a group -> must stay together)
    for s in range(12):
        for _f in range(5):
            rows.append({"source_dataset": "ai4",
                         "source_group_id": f"ai4_site{s}"})
    # sctd: 100 independent frames
    for i in range(100):
        rows.append({"source_dataset": "sctd", "source_group_id": f"sctd_{i}"})
    idx._assign_splits(rows, seed=1337)

    # R5: no group across splits
    by_group: dict[str, set[str]] = {}
    for r in rows:
        by_group.setdefault(r["source_group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values()), "group leaked across splits!"

    # every source appears in every split (per-source 70/15/15)
    for src in ("ai4", "sctd"):
        splits = {r["split"] for r in rows if r["source_dataset"] == src}
        assert splits == {"train", "val", "test"}, f"{src} missing a split: {splits}"


# ---- validate real manifests when present ----

def _real_manifests():
    return sorted(p for p in (ROOT / "data" / "manifests").glob("*.csv")
                  if not p.name.startswith("_"))


@pytest.mark.skipif(not _real_manifests(), reason="no manifests yet (T0.2 AWAITING)")
def test_real_manifests_no_leakage():
    """R5: no source_group_id crosses splits. Reads only CSV columns — CI-safe
    (never touches the image files, which are gitignored)."""
    for man in _real_manifests():
        rows = list(csv.DictReader(man.open()))
        by_group: dict[str, set[str]] = {}
        for r in rows:
            by_group.setdefault(r["source_group_id"], set()).add(r["split"])
        for g, s in by_group.items():
            assert len(s) == 1, f"{man.name}: group {g} leaked across {s}"


@pytest.mark.skipif(not _real_manifests(), reason="no manifests yet (T0.2 AWAITING)")
def test_real_manifests_sha256_integrity():
    """sha256 matches on-disk bytes. Skips rows whose files aren't present (images are
    gitignored, so this is a full check locally and a no-op in CI — reported, not silent)."""
    for man in _real_manifests():
        rows = list(csv.DictReader(man.open()))
        checked = skipped = 0
        for r in rows:
            p = Path(r["image_path"])
            if not p.exists():
                skipped += 1
                continue
            assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"], f"{man.name}: {p}"
            checked += 1
        print(f"{man.name}: sha256 checked {checked}, skipped {skipped} (files absent)")


_DET = ROOT / "data" / "manifests" / "detection.csv"
_SEG = ROOT / "data" / "manifests" / "segmentation.csv"


@pytest.mark.skipif(not (_DET.exists() and _SEG.exists()), reason="manifests absent")
def test_shared_ghost_net_split_consistent_across_tasks():
    """The synth_net ghost_net images live in BOTH detection and segmentation. A stem in
    detection-test must NOT be in segmentation-train, or the combined-system T8.1 e2e run
    leaks the novelty class across tasks. (Holds because stem sets match + split is
    deterministic — this test guards against that ever drifting.)"""
    def synth_splits(man):
        return {r["source_group_id"]: r["split"] for r in csv.DictReader(man.open())
                if r["source_dataset"] == "synth_net"}
    d, s = synth_splits(_DET), synth_splits(_SEG)
    shared = d.keys() & s.keys()
    disagree = [k for k in shared if d[k] != s[k]]
    assert not disagree, f"{len(disagree)} synth_net ids split differently across tasks"
