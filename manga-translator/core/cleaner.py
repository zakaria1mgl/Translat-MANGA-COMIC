"""
Stage 3 — Text cleaning & normalization (pure Python, no models).

- ALL-CAPS -> sentence case (translation models degrade on all-caps input).
- Comic punctuation ("!!", "!?", "...") preserved but normalized.
- SFX/onomatopoeia detection ("BOOM", "CRASH") — tagged for transliteration
  instead of literal machine translation.
- OCR artifact spell-correction via symspellpy (deterministic, no model).
"""

from __future__ import annotations

import logging
import re

from core.detector import Region

logger = logging.getLogger("manga_translator.cleaner")

# Common comic onomatopoeia — matched as whole (short, loud) tokens.
SFX_WORDS = {
    "boom", "crash", "bang", "pow", "wham", "thud", "slam", "smash", "whoosh",
    "swoosh", "zoom", "crack", "snap", "pop", "buzz", "ring", "beep", "click",
    "clang", "thump", "splash", "drip", "roar", "growl", "hiss", "screech",
    "vroom", "kaboom", "blam", "zap", "swish", "clank", "creak", "knock",
    "tap", "stomp", "crunch", "rip", "whack", "smack", "slap", "punch",
    "argh", "aargh", "gah", "ugh", "hmph", "grr", "tch", "heh", "haha",
    "hahaha", "kyaa", "gyaa", "waah", "huff", "gasp", "sigh", "sob", "sniff",
}

_symspell = None


def _get_symspell():
    global _symspell
    if _symspell is None:
        try:
            import importlib.resources

            from symspellpy import SymSpell

            sym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
            dict_path = str(
                importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            )
            sym.load_dictionary(dict_path, term_index=0, count_index=1)
            _symspell = sym
        except Exception as exc:
            logger.warning("SymSpell unavailable (%s); skipping spell correction", exc)
            _symspell = False
    return _symspell or None


def is_sfx(text: str) -> bool:
    """SFX = short, shouty, and made of known onomatopoeia (or one loud word)."""
    stripped = re.sub(r"[^A-Za-z\s]", "", text).strip()
    if not stripped:
        return False
    words = stripped.lower().split()
    if len(words) > 3:
        return False
    if all(w in SFX_WORDS for w in words):
        return True
    # Single ALL-CAPS word with heavy punctuation reads as SFX too.
    return (
        len(words) == 1
        and text.strip() == text.strip().upper()
        and len(words[0]) <= 10
        and bool(re.search(r"[!]{1,}", text))
    )


def _fix_all_caps(text: str) -> str:
    """Comic lettering is ALL CAPS; convert to sentence case for the translator."""
    letters = re.sub(r"[^A-Za-z]", "", text)
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.9:
        return text  # mixed case already — leave intentional emphasis alone
    lowered = text.lower()
    # Capitalize sentence starts.
    def cap(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    result = re.sub(r"(^|[.!?]\s+)([a-z])", cap, lowered)
    result = re.sub(r"\bi\b", "I", result)  # pronoun I
    return result


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # "happen- ing" -> "happening"
    text = re.sub(r"\.{4,}", "...", text)       # "....." -> "..."
    text = re.sub(r"!{3,}", "!!", text)         # "!!!!"  -> "!!"
    text = re.sub(r"\?{3,}", "??", text)
    text = re.sub(r"‘|’", "'", text)
    text = re.sub(r"“|”", '"', text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # no space before punctuation
    return text.strip()


def _spell_correct(text: str) -> str:
    sym = _get_symspell()
    if sym is None:
        return text
    out: list[str] = []
    for word in text.split():
        core = re.sub(r"[^A-Za-z']", "", word)
        if len(core) < 4 or core.lower() in SFX_WORDS:
            out.append(word)
            continue
        suggestions = sym.lookup(core.lower(), verbosity=0, max_edit_distance=1)
        if suggestions and suggestions[0].term != core.lower() and suggestions[0].distance == 1:
            corrected = suggestions[0].term
            if core[0].isupper():
                corrected = corrected.capitalize()
            out.append(word.replace(core, corrected))
        else:
            out.append(word)
    return " ".join(out)


def clean_region(reg: Region) -> Region:
    raw = reg.text_en_raw
    reg.is_sfx = reg.is_sfx or is_sfx(raw)
    if reg.is_sfx:
        reg.text_en = _normalize_punctuation(raw)  # keep SFX loud; no case fix
        return reg
    text = _normalize_punctuation(raw)
    text = _spell_correct(text)
    text = _fix_all_caps(text)
    reg.text_en = text
    return reg


def clean_page(regions: list[Region]) -> list[Region]:
    for reg in regions:
        clean_region(reg)
    return regions
