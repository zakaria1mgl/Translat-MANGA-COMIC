"""Shared image operations: crops, masks, NMS, color sampling."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def crop_with_padding(
    img: np.ndarray, box: tuple[int, int, int, int], pad: int = 10
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop box (x1, y1, x2, y2) with padding, clamped to image bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def nms_merge(regions: list, iou_threshold: float = 0.5) -> list:
    """
    Non-Maximum Suppression across detections from multiple detectors.
    `regions` are objects with .box (x1,y1,x2,y2) and .confidence.
    Keeps the highest-confidence region; a segmentation mask wins over none.
    """
    regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
    kept: list = []
    for reg in regions:
        duplicate = None
        for k in kept:
            if iou(reg.box, k.box) >= iou_threshold:
                duplicate = k
                break
        if duplicate is None:
            kept.append(reg)
        elif duplicate.mask is None and reg.mask is not None:
            duplicate.mask = reg.mask  # steal the better mask
    return kept


def text_pixel_mask(crop_bgr: np.ndarray, dilate_px: int = 4) -> np.ndarray:
    """
    Estimate which pixels inside a bubble crop are TEXT (high contrast vs fill),
    then dilate to catch anti-aliased edges. Returns uint8 mask (255 = text).
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu splits text ink from bubble fill robustly for both dark-on-light
    # and light-on-dark lettering.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure text (minority class) is white in the mask.
    if np.count_nonzero(th) > th.size // 2:
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    return cv2.dilate(th, kernel, iterations=1)


def dominant_fill_color(crop_bgr: np.ndarray, text_mask: np.ndarray) -> tuple[int, int, int]:
    """Median color of NON-text pixels — the bubble's fill color (BGR)."""
    inv = cv2.bitwise_not(text_mask)
    pixels = crop_bgr[inv > 0]
    if len(pixels) == 0:
        return (255, 255, 255)
    med = np.median(pixels, axis=0).astype(int)
    return int(med[0]), int(med[1]), int(med[2])


def dominant_text_color(crop_bgr: np.ndarray, text_mask: np.ndarray) -> tuple[int, int, int]:
    """Median color of text pixels (RGB, for PIL rendering)."""
    pixels = crop_bgr[text_mask > 0]
    if len(pixels) == 0:
        return (0, 0, 0)
    med = np.median(pixels, axis=0).astype(int)
    return int(med[2]), int(med[1]), int(med[0])  # BGR -> RGB


def is_solid_fill(crop_bgr: np.ndarray, text_mask: np.ndarray, std_threshold: float = 12.0) -> bool:
    """True if the non-text area of the crop is a (near-)solid color."""
    inv = cv2.bitwise_not(text_mask)
    pixels = crop_bgr[inv > 0]
    if len(pixels) < 20:
        return False
    return float(np.std(pixels, axis=0).mean()) < std_threshold


def estimate_text_angle(crop_bgr: np.ndarray, text_mask: np.ndarray) -> float:
    """Estimate rotation angle (degrees) of text from its pixel mask (for SFX)."""
    points = cv2.findNonZero(text_mask)
    if points is None or len(points) < 20:
        return 0.0
    rect = cv2.minAreaRect(points)
    angle = rect[2]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle)
