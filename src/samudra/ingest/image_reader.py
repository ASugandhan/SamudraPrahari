"""PNG/JPG → SonarFrame (degraded mode).

An image upload has no navigation and no altitude, so those fields stay None
(Rules R1, R2) and mode is "degraded".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from samudra.common.types import SonarFrame


def read_image(path: str | Path) -> SonarFrame:
    p = Path(path)
    img = Image.open(p).convert("L")  # grayscale
    pixels = np.asarray(img, dtype=np.uint8)
    return SonarFrame(
        pixels=pixels,
        mode="degraded",
        source_file=str(p),
        ping_nav=None,
        altitude_m=None,
        altitude_px=None,
        slant_res_m_per_px=None,
        along_res_m_per_px=None,
        frequency_khz=None,
    )
