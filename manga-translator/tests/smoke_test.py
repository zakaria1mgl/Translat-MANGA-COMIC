"""
Smoke test for the model-free stages: cleaning (Stage 3) and Arabic
typesetting (Stage 6). Run with the lightweight deps only:

    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.cleaner import clean_region, is_sfx
from core.detector import Region
from core.typesetter import typeset_page
from utils.arabic_text import has_raqm, is_arabic, prepare_for_render, wrap_rtl


def test_cleaner() -> None:
    reg = Region(box=(0, 0, 100, 100), confidence=0.9, kind="bubble")
    reg.text_en_raw = "I CAN'T BELIEVE  THIS IS HAPPEN- ING!!!!"
    clean_region(reg)
    assert reg.text_en, "cleaner produced empty text"
    assert "!!!!" not in reg.text_en, f"punctuation not normalized: {reg.text_en!r}"
    assert not reg.text_en.isupper(), f"ALL-CAPS not fixed: {reg.text_en!r}"

    assert is_sfx("BOOM!") is True
    assert is_sfx("CRASH") is True
    assert is_sfx("Hello there, how are you today?") is False
    print("cleaner: OK ->", repr(reg.text_en))


def test_arabic_utils() -> None:
    sample = "مرحباً بالعالم"
    assert is_arabic(sample)
    assert not is_arabic("hello world")
    rendered = prepare_for_render(sample)
    assert rendered, "prepare_for_render returned empty"
    lines = wrap_rtl("هذا نص عربي طويل جداً يجب أن يلتف على عدة أسطر", len, 15)
    assert len(lines) >= 2, f"RTL wrap failed: {lines}"
    print(f"arabic_utils: OK (raqm={has_raqm()}, wrapped into {len(lines)} lines)")


def test_typesetting() -> None:
    # White page with one solid "bubble" region.
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    reg = Region(box=(100, 80, 500, 320), confidence=0.95, kind="bubble")
    reg.text_ar = "لن أستسلم أبداً مهما حدث!"
    reg.text_color = (0, 0, 0)
    reg.solid_fill = True

    out = typeset_page(page, [reg])
    assert out.shape == page.shape
    # Text must have changed pixels inside the bubble box.
    x1, y1, x2, y2 = reg.box
    changed = int(np.count_nonzero(out[y1:y2, x1:x2] != page[y1:y2, x1:x2]))
    assert changed > 500, f"typesetting drew almost nothing ({changed} px changed)"

    import cv2

    out_path = Path(__file__).parent / "typeset_sample.png"
    cv2.imwrite(str(out_path), out)
    print(f"typesetting: OK ({changed} px drawn) -> {out_path}")


if __name__ == "__main__":
    test_cleaner()
    test_arabic_utils()
    test_typesetting()
    print("\nALL SMOKE TESTS PASSED")
