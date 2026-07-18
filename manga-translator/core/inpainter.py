"""
Stage 5 — Text removal.

Two-tier strategy:
  - Solid-fill bubbles: fill text pixels with the bubble's fill color
    (perfect result, zero artifacts, fast).
  - Free text / narration over artwork: LaMa inpainting via iopaint
    (the gold standard for text removal over art).

Mask generation: Stage-1 segmentation mask, refined by thresholding text
pixels inside each region, dilated 3-5px to catch anti-aliased edges.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from core.detector import Region, region_full_mask
from utils.device import get_device, registry
from utils.image_ops import crop_with_padding, text_pixel_mask

logger = logging.getLogger("manga_translator.inpainter")

TEXT_DILATE_PX = 4


def _load_lama():
    from iopaint.model_manager import ModelManager

    return ModelManager(name="lama", device=get_device())


def _lama_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from iopaint.schema import HDStrategy, InpaintRequest

    manager = registry.swap_in("inpainter", _load_lama)
    config = InpaintRequest(hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=64)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = manager(rgb, mask, config)
    # iopaint returns BGR
    return result.astype(np.uint8)


def build_text_mask(img_bgr: np.ndarray, reg: Region) -> np.ndarray:
    """Full-image mask of the region's TEXT pixels (255 = remove)."""
    full = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    crop, (x1, y1, x2, y2) = crop_with_padding(img_bgr, reg.box, pad=6)
    if crop.size == 0:
        return full
    local = text_pixel_mask(crop, dilate_px=TEXT_DILATE_PX)
    # Constrain to the segmentation mask when we have one (bubble interior only).
    seg = region_full_mask(reg, img_bgr.shape)[y1:y2, x1:x2]
    if seg.shape == local.shape:
        seg_dilated = cv2.dilate(
            seg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        local = cv2.bitwise_and(local, seg_dilated)
    full[y1:y2, x1:x2] = local
    return full


def _flood_fill_solid(img_bgr: np.ndarray, mask: np.ndarray, fill_color) -> None:
    """In-place: paint masked pixels with the bubble's fill color."""
    img_bgr[mask > 0] = fill_color


def remove_text(
    img_bgr: np.ndarray, regions: list[Region], quality_max: bool = True
) -> np.ndarray:
    """Return a copy of the page with all detected text removed."""
    out = img_bgr.copy()
    lama_mask = np.zeros(out.shape[:2], dtype=np.uint8)

    for reg in regions:
        mask = build_text_mask(out, reg)
        if reg.kind == "bubble" and reg.solid_fill:
            _flood_fill_solid(out, mask, reg.fill_color)
        elif quality_max:
            lama_mask = cv2.bitwise_or(lama_mask, mask)
        else:
            # Fast mode: Telea for non-bubble text.
            out = cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA)

    if quality_max and np.any(lama_mask):
        try:
            out = _lama_inpaint(out, lama_mask)
        except Exception as exc:
            logger.warning("LaMa failed (%s); falling back to Telea", exc)
            out = cv2.inpaint(out, lama_mask, 7, cv2.INPAINT_TELEA)

    return out
