"""In-memory image preparation. Nothing here writes pixels to disk.

Every image is decoded and re-encoded as JPEG, which also drops metadata such
as EXIF location tags.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Sequence

from PIL import Image

from .errors import StatesureError

MAX_PIXELS = 40_000_000
PRIMARY_SIZE = (960, 540)
STORE_SIZE = (1280, 720)
FULL_VIEW_SIZE = (1280, 720)
CROP_SIZE = (960, 720)
MIN_CROP_WIDTH = 320


class ImageError(StatesureError):
    """An image could not be decoded or processed."""


def _open(raw: bytes) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            image = Image.open(io.BytesIO(raw))
            if image.width * image.height > MAX_PIXELS:
                raise ImageError("image_too_large")
            image.load()
        except ImageError:
            raise
        except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise ImageError("invalid_image") from None
    return image.convert("RGB")


def _encode(image: Image.Image, size: tuple[int, int], *, quality: int, upscale: bool) -> bytes:
    image = image.copy()
    if upscale and image.width < MIN_CROP_WIDTH:
        scale = min(size[0] / image.width, size[1] / image.height, 2.0)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image.thumbnail(size, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def primary_view(raw: bytes) -> bytes:
    """The image sent for the first judgment."""
    return _encode(_open(raw), PRIMARY_SIZE, quality=82, upscale=False)


def stored_view(raw: bytes) -> bytes:
    """The image kept, with an expiry, for human review."""
    return _encode(_open(raw), STORE_SIZE, quality=85, upscale=False)


def verification_views(raw: bytes, roi: Sequence[tuple[float, float, float, float]]) -> list[bytes]:
    """Full frame plus one crop per normalized region, for the focused re-check."""
    image = _open(raw)
    views = [_encode(image, FULL_VIEW_SIZE, quality=90, upscale=True)]
    for x0, y0, x1, y1 in roi:
        box = (
            round(x0 * image.width),
            round(y0 * image.height),
            round(x1 * image.width),
            round(y1 * image.height),
        )
        views.append(_encode(image.crop(box), CROP_SIZE, quality=90, upscale=True))
    return views
