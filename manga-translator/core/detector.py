"""
Stage 1 — Speech bubble & text region detection.

Ensemble of two YOLOv8 detectors:
  1. kitsumed/yolov8m_seg-speech-bubble  (segmentation — exact bubble masks)
  2. ogkalu/comic-text-and-bubble-detector (free text / SFX / narration boxes)

Results are merged with NMS (IoU 0.5). Every region is classified as
'bubble', 'narration_box', or 'free_text' — each gets different treatment
downstream (flood-fill vs LaMa inpainting, font/stroke choices, etc).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from utils.device import get_device, registry
from utils.image_ops import nms_merge

logger = logging.getLogger("manga_translator.detector")

BUBBLE_SEG_REPO = "kitsumed/yolov8m_seg-speech-bubble"
BUBBLE_SEG_FILE = "model.pt"
TEXT_DET_REPO = "ogkalu/comic-text-and-bubble-detector"
TEXT_DET_FILE = "comic-text-and-bubble-detector.pt"

MIN_LONG_SIDE = 1024        # never infer below this resolution
CONF_THRESHOLD = 0.30       # low on purpose; OCR stage discards empty regions
NMS_IOU = 0.50


@dataclass
class Region:
    box: tuple[int, int, int, int]              # x1, y1, x2, y2 (full-image coords)
    confidence: float
    kind: str                                   # 'bubble' | 'narration_box' | 'free_text'
    mask: np.ndarray | None = None              # full-image uint8 mask (255 inside)
    # Filled by later stages:
    text_en: str = ""
    text_en_raw: str = ""
    ocr_confidence: float = 0.0
    text_ar: str = ""
    is_sfx: bool = False
    text_color: tuple[int, int, int] = (0, 0, 0)
    fill_color: tuple[int, int, int] = (255, 255, 255)
    solid_fill: bool = False
    angle: float = 0.0
    extra: dict = field(default_factory=dict)


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


def _load_bubble_model():
    from ultralytics import YOLO

    return YOLO(_download(BUBBLE_SEG_REPO, BUBBLE_SEG_FILE))


def _load_text_model():
    from ultralytics import YOLO

    try:
        return YOLO(_download(TEXT_DET_REPO, TEXT_DET_FILE))
    except Exception as exc:  # secondary detector is best-effort
        logger.warning("Secondary text detector unavailable: %s", exc)
        return None


def _prepare_image(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Upscale so the long side is >= MIN_LONG_SIDE. Returns (image, scale)."""
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    if long_side >= MIN_LONG_SIDE:
        return img_bgr, 1.0
    scale = MIN_LONG_SIDE / long_side
    resized = cv2.resize(
        img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4
    )
    return resized, scale


def _run_bubble_seg(model, img: np.ndarray) -> list[Region]:
    regions: list[Region] = []
    results = model.predict(
        img, conf=CONF_THRESHOLD, device=get_device(), verbose=False, retina_masks=True
    )
    for res in results:
        if res.boxes is None:
            continue
        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            mask = None
            if masks is not None and i < len(masks):
                m = (masks[i] * 255).astype(np.uint8)
                if m.shape[:2] != img.shape[:2]:
                    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = m
            regions.append(Region(box=(x1, y1, x2, y2), confidence=conf, kind="bubble", mask=mask))
    return regions


def _run_text_det(model, img: np.ndarray) -> list[Region]:
    if model is None:
        return []
    regions: list[Region] = []
    results = model.predict(img, conf=CONF_THRESHOLD, device=get_device(), verbose=False)
    for res in results:
        if res.boxes is None:
            continue
        names = res.names or {}
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = str(names.get(int(box.cls[0]), "text")).lower()
            if "bubble" in label:
                kind = "bubble"
            elif "narration" in label or "box" in label:
                kind = "narration_box"
            else:
                kind = "free_text"
            regions.append(Region(box=(x1, y1, x2, y2), confidence=conf, kind=kind))
    return regions


def _rescale_region(reg: Region, scale: float, orig_shape: tuple[int, int]) -> Region:
    if scale == 1.0:
        return reg
    inv = 1.0 / scale
    x1, y1, x2, y2 = reg.box
    reg.box = (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
    if reg.mask is not None:
        reg.mask = cv2.resize(
            reg.mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return reg


def detect_regions(img_bgr: np.ndarray) -> list[Region]:
    """Run the detector ensemble on a full page. Returns merged regions,
    sorted in comic reading order (top-to-bottom, then right-to-left for
    the eventual RTL output)."""
    orig_shape = img_bgr.shape[:2]
    img, scale = _prepare_image(img_bgr)

    bubble_model = registry.swap_in("detector_bubble", _load_bubble_model)
    text_model = registry.swap_in("detector_text", _load_text_model)

    regions = _run_bubble_seg(bubble_model, img) + _run_text_det(text_model, img)
    merged = nms_merge(regions, iou_threshold=NMS_IOU)
    merged = [_rescale_region(r, scale, orig_shape) for r in merged]

    # Reading order: row-major top-to-bottom, right-to-left within a row band.
    merged.sort(key=lambda r: (r.box[1] // 150, -r.box[0]))
    logger.info("Detected %d regions (%s)", len(merged), [r.kind for r in merged])
    return merged


def region_full_mask(reg: Region, img_shape: tuple[int, int]) -> np.ndarray:
    """Full-image mask for a region — segmentation mask if present, else its box."""
    if reg.mask is not None:
        return reg.mask
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = reg.box
    mask[y1:y2, x1:x2] = 255
    return mask
