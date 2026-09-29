"""The canonical preprocessing pipeline — the ONLY public entry point (Rule R8).

Fixed order: bottom → slant → nadir → gain → despeckle → contrast → quality → tiling.
Each step is toggleable via `enabled`, but the order never changes. There is exactly one
public function here: `run`. A test enforces both the order and the single-function rule.

    python -m samudra.preprocess.pipeline --file <sample> --save-steps out/
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from samudra.common import load_yaml
from samudra.common.logging import get_logger
from samudra.common.types import SonarFrame, Tile
from samudra.ingest import read_any

from . import bottom_tracking, contrast, despeckle, gain, nadir, quality, slant_range, tiling

log = get_logger("pipeline")

# The one true order. Editing this list is what the order test guards (R8).
STEP_ORDER = ["bottom", "slant", "nadir", "gain", "despeckle", "contrast", "quality", "tiling"]


@dataclass
class _State:
    frame: SonarFrame
    img: np.ndarray
    valid: np.ndarray
    altitude_px: np.ndarray | None = None
    tiles: list[Tile] | None = None
    steps_run: list[str] = field(default_factory=list)
    step_images: dict[str, np.ndarray] = field(default_factory=dict)


def _step_bottom(s: _State, cfg: dict) -> None:
    bottom_tracking.track_bottom(s.frame, cfg["bottom_tracking"])
    s.altitude_px = s.frame.altitude_px


def _step_slant(s: _State, cfg: dict) -> None:
    if s.altitude_px is None:
        s.altitude_px = np.zeros(s.img.shape[0], np.float32)
    s.img, s.valid = slant_range.slant_to_ground(s.img, s.altitude_px, cfg["slant_range"])


def _step_nadir(s: _State, cfg: dict) -> None:
    alt = s.altitude_px if s.altitude_px is not None else np.zeros(s.img.shape[0], np.float32)
    s.img, s.valid = nadir.mask_nadir(s.img, alt, cfg["nadir"], valid_mask=s.valid)


def _step_gain(s: _State, cfg: dict) -> None:
    s.img = gain.normalise_gain(s.img, cfg["gain"], s.valid)


def _step_despeckle(s: _State, cfg: dict) -> None:
    s.img = despeckle.despeckle(s.img, cfg["despeckle"])


def _step_contrast(s: _State, cfg: dict) -> None:
    s.img = contrast.contrast_normalise(s.img, cfg["contrast"])


def _step_quality(s: _State, cfg: dict) -> None:
    # per-tile quality gate: tile the (contrast) image, score + flag each tile.
    if s.tiles is None:
        s.tiles = tiling.tile_image(s.img, cfg["tiling"], s.valid)
    qcfg = load_yaml("configs/quality.yaml")
    for t in s.tiles:
        score, _ = quality.quality_score(t.pixels, qcfg)
        t.quality_score = score
        t.quality_flag = quality.quality_flag(score, qcfg)


def _step_tiling(s: _State, cfg: dict) -> None:
    if s.tiles is None:  # quality was disabled — still emit tiles
        s.tiles = tiling.tile_image(s.img, cfg["tiling"], s.valid)


_STEPS = {
    "bottom": _step_bottom,
    "slant": _step_slant,
    "nadir": _step_nadir,
    "gain": _step_gain,
    "despeckle": _step_despeckle,
    "contrast": _step_contrast,
    "quality": _step_quality,
    "tiling": _step_tiling,
}


@dataclass
class PipelineResult:
    frame: SonarFrame
    tiles: list[Tile]
    steps_run: list[str]
    timings: dict[str, float]
    image: np.ndarray  # final full-image (pre-tiling) result


def run(
    source: str | Path | SonarFrame,
    cfg: dict | None = None,
    enabled: dict[str, bool] | None = None,
    save_steps: str | Path | None = None,
) -> PipelineResult:
    """Run the canonical pipeline. `source` is a file path or a SonarFrame."""
    cfg = cfg or load_yaml("configs/preprocess.yaml")
    frame = source if isinstance(source, SonarFrame) else read_any(str(source))
    s = _State(frame=frame, img=frame.pixels.astype(np.float32),
               valid=np.ones(frame.pixels.shape, bool))

    timings: dict[str, float] = {}
    for name in STEP_ORDER:                       # order is fixed here (R8)
        if enabled is not None and not enabled.get(name, True):
            continue
        t0 = time.perf_counter()
        _STEPS[name](s, cfg)
        timings[name] = time.perf_counter() - t0
        s.steps_run.append(name)
        s.step_images[name] = s.img.copy()

    if save_steps:
        _save_steps(s, save_steps)

    frame.pixels = s.img if s.img.dtype == np.uint8 else s.img.astype(np.uint8)
    frame.valid_mask = s.valid
    return PipelineResult(frame=frame, tiles=s.tiles or [], steps_run=s.steps_run,
                          timings=timings, image=s.img)


def _save_steps(s: _State, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(s.steps_run):
        img = s.step_images[name]
        a = np.clip(img, 0, None)
        a = (255 * (a - a.min()) / (np.ptp(a) + 1e-6)).astype(np.uint8)
        Image.fromarray(a).save(out / f"{i}_{name}.png")
    log.info("saved %d step PNGs to %s", len(s.steps_run), out)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--save-steps", default=None)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run(a.file, save_steps=a.save_steps)
    dt = time.perf_counter() - t0
    print(f"steps: {res.steps_run}")
    print(f"tiles: {len(res.tiles)}  final image: {res.image.shape}")
    print(f"per-step timings (s): {{{', '.join(f'{k}:{v:.3f}' for k, v in res.timings.items())}}}")
    print(f"total: {dt:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
