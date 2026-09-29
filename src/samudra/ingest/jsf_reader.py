"""EdgeTech JSF → SonarFrame.

JSF is a stream of messages, each prefixed by a 16-byte header:
    uint16 start_marker (0x1601), uint8 version, uint8 session, uint16 msg_type,
    uint8 cmd, uint8 subsystem, uint8 channel, uint8 seq, uint16 reserved,
    uint32 size_following.
Sonar data messages are type 80. We frame the stream (well-defined) and decode the
sample payload per its data-format code.

Navigation/altitude live at fixed offsets in the 240-byte sonar-data header, but those
offsets are NOT validated here against a real JSF, so we leave nav/altitude = None
rather than risk fabricating coordinates (Rules R1, R2). Wire them up once a sample JSF
is available and verified. Sample count and pixels are safe to extract.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from samudra.common.logging import get_logger
from samudra.common.types import SonarFrame

log = get_logger(__name__)

_HDR = struct.Struct("<HBBHBBBBHI")  # 16 bytes
_START = 0x1601
_SONAR_DATA = 80

# JSF data-format codes -> numpy dtype for the sample payload
_FMT = {0: np.int16, 1: np.int16, 2: np.int32, 3: np.int32, 9: np.uint16}


def read_jsf(path: str | Path) -> SonarFrame:
    p = Path(path)
    raw = p.read_bytes()
    rows: list[np.ndarray] = []
    off, n = 0, len(raw)
    while off + _HDR.size <= n:
        (marker, _ver, _sess, mtype, _cmd, _sub, _ch, _seq, _res, size) = _HDR.unpack_from(raw, off)
        if marker != _START:
            off += 1  # resync
            continue
        body_start = off + _HDR.size
        body_end = body_start + size
        if body_end > n:
            break
        if mtype == _SONAR_DATA:
            rows.append(_decode_sonar_msg(raw[body_start:body_end]))
        off = body_end

    if not rows:
        raise ValueError(f"no sonar-data (type 80) messages in {p}")

    w = max(r.size for r in rows)
    pixels = np.zeros((len(rows), w), np.float32)
    for i, r in enumerate(rows):
        pixels[i, : r.size] = r
    pixels = _to_uint8(pixels)

    log.warning("JSF nav/altitude parsing is unvalidated; leaving nav/altitude=None (R1/R2)")
    return SonarFrame(
        pixels=pixels,
        mode="full",
        source_file=str(p),
        ping_nav=None,
        altitude_m=None,
        slant_res_m_per_px=None,
    )


def _decode_sonar_msg(body: bytes) -> np.ndarray:
    """Decode one type-80 message: 240-byte header then samples."""
    if len(body) < 240:
        return np.zeros(0, np.float32)
    n_samples = struct.unpack_from("<H", body, 114)[0]  # NumberOfSamples (uint16)
    data_fmt = struct.unpack_from("<H", body, 126)[0]  # DataFormat
    dtype = _FMT.get(data_fmt, np.int16)
    payload = body[240:]
    arr = np.frombuffer(payload, dtype=dtype, count=min(n_samples, len(payload) // np.dtype(dtype).itemsize))
    return arr.astype(np.float32)


def _to_uint8(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
