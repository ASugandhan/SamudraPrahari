"""Anomaly map → candidate regions (T2.2).

Threshold the smoothed reconstruction-error map, morphological cleanup, connected
components, drop tiny blobs (min area in m² when resolution is known, else px), respect
the valid_mask. Each surviving region becomes a Candidate with its box, polygon, and
peak + mean anomaly score.
"""

from __future__ import annotations

import cv2
import numpy as np

from samudra.common.types import Candidate


def _min_area_px(cfg: dict, slant_res: float | None, along_res: float | None) -> float:
    if slant_res and along_res:
        px_area_m2 = slant_res * along_res
        return float(cfg.get("min_area_m2", 0.25)) / max(px_area_m2, 1e-9)
    return float(cfg.get("min_area_px", 64))


def extract_candidates(
    anomaly_map: np.ndarray,
    cfg: dict,
    valid_mask: np.ndarray | None = None,
    tile_id: str = "tile",
    threshold: float | None = None,
    slant_res_m_per_px: float | None = None,
    along_res_m_per_px: float | None = None,
) -> list[Candidate]:
    thr = float(cfg["threshold"]) if threshold is None else float(threshold)
    amap = anomaly_map.astype(np.float32)
    h, w = amap.shape

    scored = amap.copy()
    if valid_mask is not None:
        vm = valid_mask
        if vm.shape != amap.shape:
            vm = cv2.resize(vm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        scored = np.where(vm.astype(bool), amap, 0.0)  # suppress invalid regions (R1/R2)

    binmask = (scored >= thr).astype(np.uint8)
    k = int(cfg.get("morph_kernel", 5))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, kernel)
        binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, kernel)

    min_area = _min_area_px(cfg, slant_res_m_per_px, along_res_m_per_px)
    # a "candidate" spanning most of the tile is not a localisation, not a detection
    max_area = float(cfg.get("max_area_frac", 1.0)) * h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)

    cands: list[Candidate] = []
    for i in range(1, n):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        comp = labels == i
        vals = amap[comp]
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        poly = [(int(p[0][0]), int(p[0][1])) for p in contours[0]] if contours else None
        cands.append(Candidate(
            box=(x, y, x + bw, y + bh),
            tile_id=tile_id,
            origin="anomaly",
            anomaly_score=float(vals.max()),   # peak
            anomaly_mean=float(vals.mean()),   # mean
            polygon=poly,
        ))
    return cands
