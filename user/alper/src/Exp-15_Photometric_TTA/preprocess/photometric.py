"""Photometric preprocessing variants for Exp-15.

Each variant exposes a `transform(img: PIL.Image) -> PIL.Image` callable
that is applied AFTER center-crop-square + resize-to-518 and BEFORE
JPEG85 compression in the test-tile preprocessing chain. This insertion
point matches where RandAugment operates on ConvNeXt training inputs.

Three variants:
    unsharp     PIL UnsharpMask (radius=2.0, percent=120, threshold=3)
    clahe_lab   CLAHE (clip=2.0, tile=8x8) on L channel of LAB
    saturation  ImageEnhance.Color(1.4) then Contrast(1.15)

cv2 is required for clahe_lab. If unavailable, falls back to per-channel
PIL.ImageOps.equalize (less local but no extra dependency).
"""
from __future__ import annotations

from typing import Callable, Dict

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
    import numpy as _np
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


def _identity(img: Image.Image) -> Image.Image:
    return img


def _unsharp(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=120, threshold=3))


def _clahe_lab(img: Image.Image) -> Image.Image:
    if not _HAS_CV2:
        # Fallback: per-channel histogram equalization. Less local but no cv2 dep.
        return ImageOps.equalize(img.convert("RGB"))
    arr = _np.array(img.convert("RGB"))[:, :, ::-1]  # RGB->BGR for cv2
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return Image.fromarray(bgr[:, :, ::-1])


def _saturation(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Color(img.convert("RGB")).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return img


VARIANTS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "none":       _identity,
    "unsharp":    _unsharp,
    "clahe_lab":  _clahe_lab,
    "saturation": _saturation,
}


def transform(img: Image.Image, variant: str = "none") -> Image.Image:
    """Apply a named photometric variant. Unknown variant raises ValueError."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown photometric variant: {variant}. "
            f"Choices: {sorted(VARIANTS.keys())}"
        )
    return VARIANTS[variant](img)
