#!/usr/bin/env python
"""Bottom-tracking MAE vs recorded altitude (T1.2).

With --file <sample.xtf>: tracks altitude and compares to the file's recorded altitude.
With no file: runs on a synthetic waterfall with a known altitude ramp and reports MAE
(the only evidence available until a real sample XTF is fetched).

    python scripts/eval/bottom_tracking_eval.py
    python scripts/eval/bottom_tracking_eval.py --file data/raw/.../survey.xtf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "preprocess"))
from test_bottom_tracking import make_waterfall  # reuse the generator  # noqa: E402

from samudra.common import load_yaml  # noqa: E402
from samudra.preprocess.bottom_tracking import track_bottom, track_bottom_array  # noqa: E402

CFG = load_yaml("configs/preprocess.yaml")["bottom_tracking"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    if a.file:
        from samudra.ingest import read_any

        frame = read_any(a.file)
        recorded = frame.altitude_m
        track_bottom(frame, CFG)
        if recorded is None or frame.slant_res_m_per_px is None:
            print(f"[{a.file}] tracked altitude_px range "
                  f"{frame.altitude_px.min():.1f}–{frame.altitude_px.max():.1f}; "
                  "no recorded altitude to compare.")
            return 0
        tracked_m = frame.altitude_px * frame.slant_res_m_per_px
        mae = float(np.nanmean(np.abs(tracked_m - recorded)))
        pct = 100 * mae / float(np.nanmean(recorded))
        print(f"[{a.file}] MAE = {mae:.3f} m ({pct:.1f}% of mean recorded altitude)")
        return 0

    alt = np.linspace(28, 40, 120).astype(np.float32)
    tracked = track_bottom_array(make_waterfall(alt), CFG)
    mae = float(np.mean(np.abs(tracked - alt)))
    print(f"[synthetic] known altitude 28→40 px; tracked MAE = {mae:.3f} px "
          f"({100 * mae / alt.mean():.1f}% of mean) — target ≤ 10%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
