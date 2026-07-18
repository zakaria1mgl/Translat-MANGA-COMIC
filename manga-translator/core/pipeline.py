"""
Pipeline orchestrator — glues the six stages together and exposes the
two-phase flow the UI needs:

  Phase A (analyze) : detect -> OCR -> clean -> translate
                      -> returns editable [# , English, Arabic] rows
  Phase B (render)  : (after human review) inpaint -> typeset -> final image

Models are loaded lazily and heavy ones are swapped (see utils.device).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from core.cleaner import clean_page
from core.detector import Region, detect_regions
from core.inpainter import remove_text
from core.ocr import ocr_page
from core.translator import Glossary, translate_page
from core.typesetter import typeset_page
from utils.device import free_memory

logger = logging.getLogger("manga_translator.pipeline")


@dataclass
class PageJob:
    """One page moving through the pipeline."""

    name: str
    original_bgr: np.ndarray
    regions: list[Region] | None = None
    clean_bgr: np.ndarray | None = None
    output_bgr: np.ndarray | None = None

    @property
    def analyzed(self) -> bool:
        return self.regions is not None


def analyze_page(
    img_bgr: np.ndarray,
    glossary: Glossary,
    quality_max: bool = True,
    name: str = "page",
) -> PageJob:
    """Phase A: detection -> OCR -> cleaning -> translation."""
    job = PageJob(name=name, original_bgr=img_bgr)

    logger.info("[%s] Stage 1: detection", name)
    regions = detect_regions(img_bgr)

    logger.info("[%s] Stage 2: OCR (%d regions)", name, len(regions))
    regions = ocr_page(img_bgr, regions, quality_max=quality_max)

    logger.info("[%s] Stage 3: cleaning", name)
    regions = clean_page(regions)

    logger.info("[%s] Stage 4: translation (%d texts)", name, len(regions))
    regions = translate_page(regions, glossary)

    job.regions = regions
    free_memory()
    return job


def review_rows(job: PageJob) -> list[list]:
    """Editable rows for the human-review table: [#, English, Arabic]."""
    if not job.regions:
        return []
    return [[i + 1, r.text_en, r.text_ar] for i, r in enumerate(job.regions)]


def apply_review(job: PageJob, rows: list[list]) -> None:
    """Write the user's corrected Arabic (and English) back into the regions."""
    if not job.regions:
        return
    for row in rows or []:
        try:
            idx = int(row[0]) - 1
        except (ValueError, TypeError, IndexError):
            continue
        if 0 <= idx < len(job.regions):
            if len(row) >= 2 and str(row[1]).strip():
                job.regions[idx].text_en = str(row[1]).strip()
            if len(row) >= 3 and str(row[2]).strip():
                job.regions[idx].text_ar = str(row[2]).strip()


def render_page(job: PageJob, quality_max: bool = True) -> np.ndarray:
    """Phase B: inpainting -> typesetting."""
    assert job.regions is not None, "analyze_page must run first"

    logger.info("[%s] Stage 5: inpainting", job.name)
    job.clean_bgr = remove_text(job.original_bgr, job.regions, quality_max=quality_max)

    logger.info("[%s] Stage 6: typesetting", job.name)
    job.output_bgr = typeset_page(job.clean_bgr, job.regions)

    free_memory()
    return job.output_bgr
