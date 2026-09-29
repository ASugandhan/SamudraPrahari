"""Core dataclasses shared across the pipeline.

Fields follow the master build prompt (T1.1 SonarFrame, T1.8 Tile, T2.4 Candidate,
T5.2 Contact). Unknown physical quantities are `None` — never fabricated (Rules R1, R2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PingNav:
    """Per-ping navigation. Any field may be None when the source lacks it (R1)."""

    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    time_s: float | None = None


@dataclass
class SonarFrame:
    """One sonar image plus per-ping metadata (T1.1)."""

    pixels: np.ndarray  # H×W, port|starboard layout
    mode: str  # "full" | "degraded"
    source_file: str
    ping_nav: list[PingNav] | None = None  # per-ping nav, or None (R1)
    altitude_m: np.ndarray | None = None  # per-ping altitude in metres, or None (R2)
    altitude_px: np.ndarray | None = None  # per-ping altitude in pixels, or None
    altitude_source: str | None = None  # "file" | "bottom_tracking" | "image_estimate"
    slant_res_m_per_px: float | None = None
    along_res_m_per_px: float | None = None
    frequency_khz: float | None = None
    valid_mask: np.ndarray | None = None  # bool, True = usable pixel

    def __post_init__(self) -> None:
        if self.mode not in ("full", "degraded"):
            raise ValueError(f"mode must be full|degraded, got {self.mode!r}")
        if self.pixels.ndim != 2:
            raise ValueError(f"pixels must be 2-D H×W, got shape {self.pixels.shape}")

    @property
    def n_pings(self) -> int:
        return self.pixels.shape[0]


@dataclass
class Tile:
    """A tile carved from a SonarFrame, remembering where it came from (T1.8)."""

    tile_id: str
    pixels: np.ndarray
    row_off: int  # top-left offset in the full image
    col_off: int
    valid_mask: np.ndarray | None = None
    quality_score: float | None = None
    quality_flag: str | None = None  # "ok" | "low"

    def to_full(self, r: float, c: float) -> tuple[float, float]:
        """Map a tile-local (row, col) back to full-image coordinates."""
        return (r + self.row_off, c + self.col_off)


@dataclass
class Candidate:
    """A region of interest from Stage A (T2.4)."""

    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in full-image coords
    tile_id: str
    origin: str  # "yolo" | "anomaly" | "both"
    anomaly_score: float = 0.0  # peak reconstruction error in the region (T2.2)
    anomaly_mean: float | None = None  # mean reconstruction error in the region (T2.2)
    polygon: list[tuple[int, int]] | None = None
    yolo_class: str | None = None
    yolo_conf: float | None = None


@dataclass
class Contact:
    """A finalised detection with provenance (T5.2)."""

    id: str
    decision: str  # "KNOWN:<class>" | "UNKNOWN"
    lat: float | None = None
    lon: float | None = None
    geo_status: str = "no_nav"  # "geo" | "no_nav" | "unknown_location"
    class_conf: float | None = None
    calibrated_conf: float | None = None
    anomaly_score: float | None = None
    shadow_score: float | None = None
    match_distance: float | None = None
    area_m2: float | None = None
    dims: dict | None = None
    height_m_estimate: float | None = None
    chart_status: str | None = None  # "charted:<id>" | "new" | "unknown_location"
    quality_flag: str | None = None
    chip_path: str | None = None
    provenance: dict = field(default_factory=dict)
