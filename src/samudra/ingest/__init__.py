from pathlib import Path

from samudra.common.types import SonarFrame

from .image_reader import read_image
from .jsf_reader import read_jsf
from .xtf_reader import read_xtf

__all__ = ["read_image", "read_xtf", "read_jsf", "read_any"]

_XTF = {".xtf"}
_JSF = {".jsf"}
_IMG = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_any(path: str) -> SonarFrame:
    """Dispatch to the right reader by extension."""
    suf = Path(path).suffix.lower()
    if suf in _XTF:
        return read_xtf(path)
    if suf in _JSF:
        return read_jsf(path)
    if suf in _IMG:
        return read_image(path)
    raise ValueError(f"unsupported file type: {suf}")
