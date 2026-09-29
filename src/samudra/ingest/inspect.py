"""Inspect a sonar file: `python -m samudra.ingest.inspect <file>` (T1.1 VERIFY).

Prints ping count, whether nav is present, and the altitude range.
"""

from __future__ import annotations

import sys

import numpy as np

from . import read_any


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m samudra.ingest.inspect <file.xtf|.jsf|.png>")
        return 1
    f = read_any(argv[0])
    nav_present = f.ping_nav is not None and any(
        p.lat is not None for p in f.ping_nav
    )
    print(f"source      : {f.source_file}")
    print(f"mode        : {f.mode}")
    print(f"pixels      : {f.pixels.shape} dtype={f.pixels.dtype}")
    print(f"n_pings     : {f.n_pings}")
    print(f"nav present : {nav_present}")
    if f.altitude_m is not None:
        a = f.altitude_m[~np.isnan(f.altitude_m)]
        rng = (float(a.min()), float(a.max())) if a.size else None
        print(f"altitude_m  : range={rng} source={f.altitude_source}")
    else:
        print("altitude_m  : None")
    print(f"slant_res   : {f.slant_res_m_per_px}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
