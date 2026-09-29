"""Smoke test: package imports and core types round-trip (T0.1)."""

import numpy as np

import samudra
from samudra.common import Contact, SonarFrame, Tile, config_hash, stamp


def test_import_version():
    assert samudra.__version__


def test_sonarframe_degraded():
    f = SonarFrame(pixels=np.zeros((8, 8), np.uint8), mode="degraded", source_file="x.png")
    # degraded mode never invents nav / altitude (R1, R2)
    assert f.ping_nav is None
    assert f.altitude_m is None
    assert f.n_pings == 8


def test_tile_roundtrip():
    t = Tile(tile_id="t0", pixels=np.zeros((4, 4), np.uint8), row_off=10, col_off=20)
    assert t.to_full(1, 2) == (11, 22)


def test_provenance_stamp():
    p = stamp("stage_a_ae", "v1", {"threshold": 0.6})
    assert p["model_name"] == "stage_a_ae"
    assert len(p["config_hash"]) == 12


def test_config_hash_stable():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_contact_defaults_no_nav():
    c = Contact(id="c1", decision="UNKNOWN")
    assert c.lat is None and c.geo_status == "no_nav"
