"""Leak-free train/val/test splitting by source group (Rule R5).

Tiles from the same survey/wreck must never straddle splits. We split whole
*groups*, not individual images, with a fixed seed for reproducibility.
"""

from __future__ import annotations

import random


def split_by_group(
    group_ids: list[str],
    seed: int = 1337,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    """Assign each unique group to exactly one of train/val/test.

    Returns {group_id: split}. Because assignment is per-group, no group can
    appear in more than one split (the R5 guarantee).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    assign: dict[str, str] = {}
    for i, g in enumerate(groups):
        if i < n_train:
            assign[g] = "train"
        elif i < n_train + n_val:
            assign[g] = "val"
        else:
            assign[g] = "test"
    return assign
