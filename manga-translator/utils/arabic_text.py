"""
Arabic RTL text handling.

Strategy:
1. If Pillow was built with libraqm (`PIL.features.check("raqm")`), render with
   `direction="rtl", language="ar"` and pass the RAW string — raqm does shaping
   and bidi natively with proper kerning.
2. Otherwise, pre-process with arabic-reshaper (contextual glyph forms) +
   python-bidi (visual reordering) and render the result LTR.
"""

from __future__ import annotations

import re
from functools import lru_cache

import arabic_reshaper
from bidi.algorithm import get_display

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

_reshaper = arabic_reshaper.ArabicReshaper(
    configuration={
        "delete_harakat": False,
        "support_ligatures": True,
        "shift_harakat_position": False,
    }
)


@lru_cache(maxsize=1)
def has_raqm() -> bool:
    try:
        from PIL import features

        return bool(features.check("raqm"))
    except Exception:
        return False


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def prepare_for_render(text: str) -> str:
    """
    Return the string exactly as it must be passed to PIL's draw.text().
    With raqm: raw text (PIL handles shaping+bidi).
    Without:   reshaped + bidi-reordered text.
    """
    if has_raqm():
        return text
    reshaped = _reshaper.reshape(text)
    return get_display(reshaped)


def render_kwargs() -> dict:
    """Extra kwargs for draw.text()/draw.textbbox() when raqm is available."""
    if has_raqm():
        return {"direction": "rtl", "language": "ar"}
    return {}


def wrap_rtl(text: str, measure, max_width: int) -> list[str]:
    """
    RTL-aware greedy word wrap.

    `measure(s) -> pixel width` must measure the *raw* (unshaped) string using
    the same font that will render it. Word order is kept logical; bidi/shaping
    is applied at render time per line.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
