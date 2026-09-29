#!/usr/bin/env python
"""Fetch the three public sonar datasets into data/raw/ (T0.2).

Run on a machine with network + disk (or on Colab). The agent does NOT run this;
it produces the raw trees that build_manifest.py then indexes.

    python scripts/data/fetch.py sctd
    python scripts/data/fetch.py klsg
    python scripts/data/fetch.py ai4shipwrecks
    python scripts/data/fetch.py all
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw")

SOURCES = {
    "sctd": {
        "kind": "git",
        "url": "https://github.com/freepoet/SCTD",
        "dest": RAW / "SCTD",
    },
    "klsg": {
        "kind": "git",
        "url": "https://github.com/YDY-andy/Sonar-dataset",
        "dest": RAW / "KLSG",
    },
    "ai4shipwrecks": {
        # Large (pixel masks + terrain extras). See project page for the download link;
        # some releases are hosted off-GitHub (Deep Blue / Zenodo). Set AI4_URL env or edit.
        "kind": "manual",
        "url": "https://umfieldrobotics.github.io/ai4shipwrecks/",
        "dest": RAW / "AI4Shipwrecks",
    },
}


def _git_clone(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} exists")
        return
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def fetch(name: str) -> None:
    src = SOURCES[name]
    if src["kind"] == "git":
        _git_clone(src["url"], src["dest"])
        print(f"[ok] {name} -> {src['dest']}")
    else:
        print(
            f"[manual] {name}: download from {src['url']} and unpack into {src['dest']}.\n"
            "         AI4Shipwrecks is distributed off-GitHub; follow the project page."
        )


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in {*SOURCES, "all"}:
        print(__doc__)
        return 1
    targets = list(SOURCES) if argv[0] == "all" else [argv[0]]
    for t in targets:
        fetch(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
