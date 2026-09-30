"""Candidate fusion: YOLO boxes ∪ AE anomaly candidates → one list (T2.4).

Overlapping YOLO/AE candidates (IoU ≥ config) merge into one `origin="both"` candidate that
keeps BOTH scores. YOLO-only stay `origin="yolo"`. AE-only become `origin="anomaly_only"` —
that set is the future UNKNOWN pool (Stage C / T4.2). Fusion is a union, so fused recall can
only be ≥ the recall of either detector alone.
"""

from __future__ import annotations

from dataclasses import replace

from samudra.common.types import Candidate


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _centre_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def fuse_candidates(
    yolo_cands: list[Candidate],
    ae_cands: list[Candidate],
    cfg: dict,
) -> list[Candidate]:
    """Union with cross-origin merge. Each AE candidate merges into at most one YOLO box
    (its best overlap ≥ merge_iou); leftover AE candidates are tagged anomaly_only."""
    iou_thr = float(cfg.get("merge_iou", 0.5))
    used_ae: set[int] = set()
    fused: list[Candidate] = []

    for y in yolo_cands:
        best_j, best_iou = -1, iou_thr
        for j, a in enumerate(ae_cands):
            if j in used_ae:
                continue
            # merge if boxes overlap enough OR the anomaly blob sits inside this detection —
            # an anomaly centred inside a known YOLO object is that object, not "unknown".
            iou = box_iou(y.box, a.box)
            if iou >= best_iou or (best_j < 0 and _centre_inside(a.box, y.box)):
                best_iou, best_j = max(iou, best_iou), j
        if best_j >= 0:
            a = ae_cands[best_j]
            used_ae.add(best_j)
            fused.append(replace(
                y, origin="both",
                anomaly_score=a.anomaly_score, anomaly_mean=a.anomaly_mean,
                polygon=y.polygon or a.polygon,
            ))
        else:
            fused.append(replace(y, origin="yolo"))

    for j, a in enumerate(ae_cands):
        if j not in used_ae:
            fused.append(replace(a, origin="anomaly_only"))  # -> UNKNOWN pool (T4.2)
    return fused
