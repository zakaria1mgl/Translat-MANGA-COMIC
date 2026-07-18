"""
Stage 6 — Arabic typesetting (the most error-prone stage).

Correctness rules:
  - libraqm (PIL feature "raqm") when available: native RTL shaping + kerning.
  - Fallback: arabic-reshaper (contextual glyphs) + python-bidi (reordering).
  - Bundled fonts (Noto Naskh Arabic primary, Cairo bold for SFX/emphasis).

Algorithm per bubble:
  1. Inscribed rectangle from the segmentation mask with 8-12% padding.
  2. Binary-search the font size with RTL-aware word wrap — the largest size
     that fits guarantees maximum readability.
  3. Center text; line spacing 1.3x (Arabic needs room for diacritics).
  4. Text color sampled from the original lettering; 2px stroke over artwork.
  5. SFX: bolder font, rotated to match the original angle.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.detector import Region, region_full_mask
from utils.arabic_text import prepare_for_render, render_kwargs, wrap_rtl
from utils.image_ops import cv_to_pil, pil_to_cv

logger = logging.getLogger("manga_translator.typesetter")

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
FONT_BODY = FONTS_DIR / "NotoNaskhArabic-Regular.ttf"
FONT_BOLD = FONTS_DIR / "Cairo-Bold.ttf"

LINE_SPACING = 1.3
PADDING_RATIO = 0.10           # 8-12% inset from bubble edges
MIN_FONT, MAX_FONT = 11, 96


def _font_path(bold: bool) -> str:
    path = FONT_BOLD if bold else FONT_BODY
    if not path.exists():
        alt = FONT_BODY if bold else FONT_BOLD
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(
            f"Arabic font not found in {FONTS_DIR}. Run scripts/download_fonts.py "
            "or `python -m app --setup` to fetch bundled fonts."
        )
    return str(path)


def _inner_rect(reg: Region, img_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Largest usable rectangle inside the region, inset by PADDING_RATIO."""
    if reg.mask is not None:
        mask = region_full_mask(reg, img_shape)
        # Erode the mask so text keeps away from bubble edges, then take the
        # bounding rect of the largest remaining component.
        x1, y1, x2, y2 = reg.box
        w, h = x2 - x1, y2 - y1
        erode_px = max(3, int(min(w, h) * PADDING_RATIO))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        eroded = cv2.erode(mask, kernel)
        points = cv2.findNonZero(eroded)
        if points is not None:
            bx, by, bw, bh = cv2.boundingRect(points)
            if bw > 10 and bh > 10:
                return bx, by, bx + bw, by + bh
    x1, y1, x2, y2 = reg.box
    pad_x = int((x2 - x1) * PADDING_RATIO)
    pad_y = int((y2 - y1) * PADDING_RATIO)
    return x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y


def _load_font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(_font_path(bold), size)
    if bold:
        try:  # Cairo ships as a variable font — select the Bold instance
            font.set_variation_by_name("Bold")
        except Exception:
            pass
    return font


def _measure(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    rendered = prepare_for_render(text)
    bbox = draw.textbbox((0, 0), rendered, font=font, **render_kwargs())
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_text(
    draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int, bold: bool
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Binary-search the largest font size whose wrapped text fits the box."""
    lo, hi = MIN_FONT, MAX_FONT
    best: tuple[ImageFont.FreeTypeFont, list[str]] | None = None
    while lo <= hi:
        size = (lo + hi) // 2
        font = _load_font(size, bold)
        lines = wrap_rtl(text, lambda s: _measure(draw, font, s)[0], box_w)
        line_h = int(size * LINE_SPACING)
        total_h = line_h * len(lines)
        widest = max((_measure(draw, font, ln)[0] for ln in lines), default=0)
        if total_h <= box_h and widest <= box_w:
            best = (font, lines)
            lo = size + 1
        else:
            hi = size - 1
    if best is None:
        font = _load_font(MIN_FONT, bold)
        best = (font, wrap_rtl(text, lambda s: _measure(draw, font, s)[0], max(box_w, 1)))
    return best


def _draw_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    stroke: bool,
) -> None:
    x1, y1, x2, y2 = rect
    line_h = int(font.size * LINE_SPACING)
    total_h = line_h * len(lines)
    y = y1 + (y2 - y1 - total_h) // 2
    kwargs = render_kwargs()
    stroke_kwargs = {"stroke_width": 2, "stroke_fill": (255, 255, 255)} if stroke else {}
    for line in lines:
        rendered = prepare_for_render(line)
        w = _measure(draw, font, line)[0]
        x = x1 + (x2 - x1 - w) // 2
        draw.text((x, y), rendered, font=font, fill=color, anchor="la", **kwargs, **stroke_kwargs)
        y += line_h


def _draw_rotated(
    img: Image.Image, reg: Region, rect: tuple[int, int, int, int]
) -> Image.Image:
    """Render SFX on a transparent layer, rotate it, and composite."""
    x1, y1, x2, y2 = rect
    w, h = max(x2 - x1, 8), max(y2 - y1, 8)
    layer = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(layer)
    font, lines = _fit_text(ldraw, reg.text_ar, w, h, bold=True)
    _draw_block(ldraw, lines, font, (w // 2, h // 2, w // 2 + w, h // 2 + h), reg.text_color, stroke=True)
    rotated = layer.rotate(reg.angle, expand=False, resample=Image.BICUBIC)
    img.paste(rotated, (x1 - w // 2, y1 - h // 2), rotated)
    return img


def typeset_page(clean_bgr: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Render Arabic text onto the text-removed page. Returns BGR image."""
    pil = cv_to_pil(clean_bgr).convert("RGB")

    for reg in regions:
        if not reg.text_ar:
            continue
        rect = _inner_rect(reg, clean_bgr.shape)
        if rect[2] - rect[0] < 8 or rect[3] - rect[1] < 8:
            logger.warning("Region too small to typeset: %s", reg.box)
            continue
        if reg.is_sfx and abs(reg.angle) > 3:
            pil = _draw_rotated(pil, reg, rect)
            continue
        draw = ImageDraw.Draw(pil)
        bold = reg.is_sfx
        over_art = reg.kind == "free_text" or (reg.kind != "bubble" and not reg.solid_fill)
        font, lines = _fit_text(draw, reg.text_ar, rect[2] - rect[0], rect[3] - rect[1], bold)
        color = reg.text_color if reg.kind != "free_text" else reg.text_color
        _draw_block(draw, lines, font, rect, color, stroke=over_art)

    return pil_to_cv(pil)
