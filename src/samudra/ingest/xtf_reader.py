"""XTF (Triton eXtended Triton Format) → SonarFrame using pyxtf.

Extracts per-ping navigation, altitude, and slant-range resolution when the file
carries them. Missing fields become None — never fabricated (Rules R1, R2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from samudra.common.types import PingNav, SonarFrame


def _first_not_none(*vals):
    for v in vals:
        if v is not None and v != 0:
            return v
    return None


def read_xtf(path: str | Path) -> SonarFrame:
    import pyxtf  # imported lazily so the package installs without pyxtf

    p = Path(path)
    _, packets = pyxtf.xtf_read(str(p))
    sonar = packets.get(pyxtf.XTFHeaderType.sonar)
    if not sonar:
        raise ValueError(f"no sonar packets in {p}")

    rows_port: list[np.ndarray] = []
    rows_stbd: list[np.ndarray] = []
    navs: list[PingNav] = []
    alts: list[float | None] = []
    slant_res: list[float | None] = []

    for ping in sonar:
        chans = ping.data  # list of channel arrays
        port = np.asarray(chans[0], dtype=np.float32) if len(chans) > 0 else None
        stbd = np.asarray(chans[1], dtype=np.float32) if len(chans) > 1 else port
        rows_port.append(port if port is not None else np.zeros(1, np.float32))
        rows_stbd.append(stbd if stbd is not None else np.zeros(1, np.float32))

        lat = _first_not_none(getattr(ping, "SensorYcoordinate", None))
        lon = _first_not_none(getattr(ping, "SensorXcoordinate", None))
        heading = _first_not_none(getattr(ping, "SensorHeading", None))
        navs.append(PingNav(lat=lat, lon=lon, heading_deg=heading, time_s=None))

        alts.append(
            _first_not_none(
                getattr(ping, "SensorPrimaryAltitude", None),
                getattr(ping, "SensorAuxAltitude", None),
            )
        )

        # slant resolution = slant range / samples, from the first channel header
        chan_hdrs = getattr(ping, "ping_chan_headers", None)
        sr = None
        if chan_hdrs:
            ch = chan_hdrs[0]
            slant_range = getattr(ch, "SlantRange", None)
            nsamp = getattr(ch, "NumSamples", None)
            if slant_range and nsamp:
                sr = float(slant_range) / float(nsamp)
        slant_res.append(sr)

    # pad channels to a common width, port reversed then starboard (nadir in centre)
    w = max(max(r.size for r in rows_port), max(r.size for r in rows_stbd))

    def _stack(rows):
        out = np.zeros((len(rows), w), np.float32)
        for i, r in enumerate(rows):
            out[i, : r.size] = r
        return out

    port_img = _stack(rows_port)[:, ::-1]
    stbd_img = _stack(rows_stbd)
    pixels = np.concatenate([port_img, stbd_img], axis=1)
    pixels = _to_uint8(pixels)

    alt_arr = _maybe_array(alts)
    return SonarFrame(
        pixels=pixels,
        mode="full",
        source_file=str(p),
        ping_nav=navs,
        altitude_m=alt_arr,
        altitude_source="file" if alt_arr is not None else None,
        slant_res_m_per_px=_median_or_none(slant_res),
        along_res_m_per_px=None,
        frequency_khz=None,
    )


def _to_uint8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return (a * 255).astype(np.uint8)


def _maybe_array(vals):
    if all(v is None for v in vals):
        return None
    filled = [float(v) if v is not None else np.nan for v in vals]
    return np.asarray(filled, dtype=np.float32)


def _median_or_none(vals):
    good = [v for v in vals if v is not None]
    return float(np.median(good)) if good else None
