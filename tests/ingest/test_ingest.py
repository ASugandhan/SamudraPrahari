"""T1.1 ingest tests: image (degraded), XTF (full, mocked pyxtf), JSF (synthetic)."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pyxtf
from PIL import Image

from samudra.ingest import read_any
from samudra.ingest.image_reader import read_image
from samudra.ingest.jsf_reader import _HDR, _START, read_jsf
from samudra.ingest.xtf_reader import read_xtf

# ---- image mode: degraded, no nav/altitude (R1, R2) ----

def test_image_degraded(tmp_path):
    p = tmp_path / "sonar.png"
    Image.new("L", (32, 16), color=120).save(p)
    f = read_image(p)
    assert f.mode == "degraded"
    assert f.pixels.shape == (16, 32) and f.pixels.dtype == np.uint8
    assert f.ping_nav is None
    assert f.altitude_m is None and f.slant_res_m_per_px is None


def test_read_any_dispatch(tmp_path):
    p = tmp_path / "x.PNG"
    Image.new("L", (4, 4)).save(p)
    assert read_any(str(p)).mode == "degraded"


# ---- XTF mode: full, nav + altitude populated (mock pyxtf.xtf_read) ----

def _fake_ping(lat, lon, heading, alt, n=8):
    port = np.linspace(0, 100, n, dtype=np.float32)
    stbd = np.linspace(50, 150, n, dtype=np.float32)
    chan = SimpleNamespace(SlantRange=30.0, NumSamples=n)
    return SimpleNamespace(
        data=[port, stbd],
        SensorYcoordinate=lat,
        SensorXcoordinate=lon,
        SensorHeading=heading,
        SensorPrimaryAltitude=alt,
        SensorAuxAltitude=0.0,
        ping_chan_headers=[chan, chan],
    )


def test_xtf_full_mode(monkeypatch):
    pings = [
        _fake_ping(12.90, 74.80, 45.0, 8.0),
        _fake_ping(12.91, 74.81, 46.0, 9.0),   # varying altitude
    ]
    monkeypatch.setattr(
        pyxtf, "xtf_read", lambda _p: (None, {pyxtf.XTFHeaderType.sonar: pings})
    )
    f = read_xtf("dummy.xtf")
    assert f.mode == "full"
    assert f.n_pings == 2
    # nav populated per ping (R1: real values, not fabricated)
    assert f.ping_nav[0].lat == 12.90 and f.ping_nav[1].lon == 74.81
    assert f.ping_nav[0].heading_deg == 45.0
    # altitude populated + varying, source flagged
    assert f.altitude_m is not None
    assert f.altitude_source == "file"
    assert list(np.round(f.altitude_m, 1)) == [8.0, 9.0]
    # slant resolution derived from SlantRange / NumSamples = 30/8
    assert abs(f.slant_res_m_per_px - 30.0 / 8) < 1e-6
    # port+starboard stacked -> width = 2 * n
    assert f.pixels.shape == (2, 16)


# ---- JSF: framing + type-80 sample decode (synthetic bytes) ----

def test_jsf_synthetic(tmp_path):
    n_samples = 4
    body = bytearray(240)
    struct.pack_into("<H", body, 114, n_samples)   # NumberOfSamples
    struct.pack_into("<H", body, 126, 0)           # DataFormat 0 -> int16
    body += struct.pack("<4h", 10, 20, 30, 40)     # 4 int16 samples
    hdr = _HDR.pack(_START, 1, 0, 80, 0, 0, 0, 0, 0, len(body))
    msg = hdr + bytes(body)
    p = tmp_path / "s.jsf"
    p.write_bytes(msg + msg)   # two pings
    f = read_jsf(p)
    assert f.mode == "full"
    assert f.n_pings == 2
    assert f.pixels.shape[1] == n_samples
    # nav intentionally None (unvalidated offsets) — never fabricated (R1)
    assert f.ping_nav is None and f.altitude_m is None
