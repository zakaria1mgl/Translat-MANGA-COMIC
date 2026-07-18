"""
Stage 2 — English OCR.

Primary : PaddleOCR v4 (en) — best open model on stylized comic lettering.
Ensemble: microsoft/trocr-large-printed on the same crop; used whenever
          PaddleOCR confidence < 0.85.

Preprocessing per region (mandatory):
  1. Crop with 10px padding.
  2. Real-ESRGAN 2-4x upscale if text height < 20px.
  3. Grayscale + adaptive threshold only when contrast is low.

Multi-line: lines inside a bubble are merged; the comic hyphenation pattern
("TRANS-" / "LATION") is collapsed.
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from core.detector import Region
from utils.device import get_device, get_dtype, has_cuda, registry
from utils.image_ops import (
    crop_with_padding,
    dominant_fill_color,
    dominant_text_color,
    estimate_text_angle,
    is_solid_fill,
    text_pixel_mask,
)

logger = logging.getLogger("manga_translator.ocr")

PADDLE_CONF_TRUST = 0.85     # below this, ask TrOCR
MIN_TEXT_HEIGHT_PX = 20      # below this, upscale the crop
LOW_CONTRAST_STD = 35.0

HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")


# ── Model loaders ────────────────────────────────────────────────────────────

def _load_paddle():
    from paddleocr import PaddleOCR

    return PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=has_cuda())


def _load_trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-printed", torch_dtype=get_dtype()
    ).to(get_device())
    model.eval()
    return processor, model


def _load_upscaler():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    return RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        half=has_cuda(),
        device=get_device(),
    )


# ── Preprocessing ────────────────────────────────────────────────────────────

def _estimate_text_height(crop: np.ndarray) -> float:
    mask = text_pixel_mask(crop, dilate_px=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 8]
    return float(np.median(heights)) if heights else 999.0


def _maybe_upscale(crop: np.ndarray, quality_max: bool) -> np.ndarray:
    if _estimate_text_height(crop) >= MIN_TEXT_HEIGHT_PX:
        return crop
    if quality_max:
        try:
            upscaler = registry.swap_in("upscaler", _load_upscaler)
            out, _ = upscaler.enhance(crop, outscale=4)
            return out
        except Exception as exc:
            logger.warning("Real-ESRGAN failed (%s); using Lanczos 4x", exc)
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)


def _maybe_threshold(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) >= LOW_CONTRAST_STD:
        return crop
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


# ── OCR engines ──────────────────────────────────────────────────────────────

def _paddle_read(crop: np.ndarray) -> tuple[str, float]:
    ocr = registry.swap_in("ocr_paddle", _load_paddle)
    result = ocr.ocr(crop, cls=True)
    lines: list[tuple[float, str, float]] = []  # (y_center, text, conf)
    for page in result or []:
        for det in page or []:
            box, (text, conf) = det
            y_center = float(np.mean([p[1] for p in box]))
            lines.append((y_center, text, float(conf)))
    if not lines:
        return "", 0.0
    lines.sort(key=lambda t: t[0])
    text = " ".join(t[1] for t in lines)
    conf = float(np.mean([t[2] for t in lines]))
    return text, conf


def _trocr_read(crop: np.ndarray) -> str:
    import torch
    from PIL import Image

    processor, model = registry.swap_in("ocr_trocr", _load_trocr)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(get_device(), dtype=get_dtype())
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=128)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _merge_lines(text: str) -> str:
    text = HYPHEN_BREAK.sub(r"\1\2", text)         # "TRANS- LATION" -> "TRANSLATION"
    return re.sub(r"\s+", " ", text).strip()


# ── Public API ───────────────────────────────────────────────────────────────

def ocr_region(img_bgr: np.ndarray, reg: Region, quality_max: bool = True) -> Region:
    """OCR a single region in place. Also records visual attributes
    (text color, fill color, solid-fill flag, text angle) for later stages."""
    crop, _ = crop_with_padding(img_bgr, reg.box, pad=10)
    if crop.size == 0:
        reg.text_en_raw, reg.ocr_confidence = "", 0.0
        return reg

    # Visual attributes from the ORIGINAL crop (before any preprocessing).
    mask = text_pixel_mask(crop)
    reg.text_color = dominant_text_color(crop, mask)
    reg.fill_color = dominant_fill_color(crop, mask)
    reg.solid_fill = is_solid_fill(crop, mask)
    if reg.kind == "free_text":
        reg.angle = estimate_text_angle(crop, mask)

    processed = _maybe_threshold(_maybe_upscale(crop, quality_max))

    text, conf = _paddle_read(processed)
    if quality_max and conf < PADDLE_CONF_TRUST:
        trocr_text = _trocr_read(processed)
        if len(trocr_text) > len(text) * 0.5:      # sanity: not a degenerate output
            logger.info("TrOCR override (paddle conf %.2f): %r -> %r", conf, text, trocr_text)
            text = trocr_text
            conf = max(conf, 0.85)

    reg.text_en_raw = _merge_lines(text)
    reg.ocr_confidence = conf
    return reg


def ocr_page(img_bgr: np.ndarray, regions: list[Region], quality_max: bool = True) -> list[Region]:
    """OCR all regions; discard detector false positives (no readable text)."""
    out: list[Region] = []
    for reg in regions:
        ocr_region(img_bgr, reg, quality_max=quality_max)
        if reg.text_en_raw and re.search(r"[A-Za-z]", reg.text_en_raw):
            out.append(reg)
        else:
            logger.info("Dropped empty region %s (%s)", reg.box, reg.kind)
    return out
