"""
Stage 4 — Translation (EN -> AR). The quality core.

Primary : CohereForAI/aya-expanse-8b (LLM, context-aware)
  - GPU : 4-bit via bitsandbytes (~6GB VRAM, fits Colab T4)
  - CPU : GGUF Q4_K_M via llama-cpp-python
Fallback: facebook/nllb-200-distilled-1.3B (eng_Latn -> arb_Arab), automatic only.

Quality techniques (all mandatory):
  1. Page-level context prompting — all bubbles of a page in ONE numbered prompt.
  2. Glossary/consistency memory injected into every prompt.
  3. MSA (فصحى) register by default.
  4. Deterministic decoding: temperature=0.3, top_p=0.9.
  5. Post-validation: count match + Arabic-script regex + one retry.
"""

from __future__ import annotations

import logging
import re

from core.detector import Region
from utils.arabic_text import is_arabic
from utils.device import get_device, has_cuda, registry

logger = logging.getLogger("manga_translator.translator")

AYA_HF_REPO = "CohereForAI/aya-expanse-8b"
AYA_GGUF_REPO = "bartowski/aya-expanse-8b-GGUF"
AYA_GGUF_FILE = "aya-expanse-8b-Q4_K_M.gguf"
NLLB_REPO = "facebook/nllb-200-distilled-1.3B"

TEMPERATURE = 0.3
TOP_P = 0.9
MAX_NEW_TOKENS = 1024

NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[).:\-]\s*(.+)$")

SYSTEM_PROMPT = (
    "You are a professional comic and manga translator. Translate English comic "
    "dialogue into natural Modern Standard Arabic (الفصحى) suitable for published "
    "Arabic comics. Maintain speaker tone, gender consistency across lines, and "
    "short punchy phrasing. Sound effects (marked [SFX]) must be transliterated or "
    "localized as Arabic onomatopoeia, never translated literally. Return ONLY the "
    "numbered Arabic translations, one per line, nothing else."
)


class Glossary:
    """Per-session dictionary: recurring names/terms -> fixed Arabic renderings."""

    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def update(self, english: str, arabic: str) -> None:
        english, arabic = english.strip(), arabic.strip()
        if english and arabic:
            self.entries[english] = arabic

    def remove(self, english: str) -> None:
        self.entries.pop(english.strip(), None)

    def as_prompt_block(self) -> str:
        if not self.entries:
            return ""
        lines = "\n".join(f"- {en} => {ar}" for en, ar in self.entries.items())
        return (
            "\nGlossary — always use EXACTLY these Arabic renderings for these "
            f"names/terms:\n{lines}\n"
        )

    def to_rows(self) -> list[list[str]]:
        return [[en, ar] for en, ar in self.entries.items()]

    def from_rows(self, rows) -> None:
        self.entries = {}
        for row in rows or []:
            if len(row) >= 2 and str(row[0]).strip() and str(row[1]).strip():
                self.entries[str(row[0]).strip()] = str(row[1]).strip()


# ── Backends ─────────────────────────────────────────────────────────────────

def _load_aya_gpu():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(AYA_HF_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        AYA_HF_REPO, quantization_config=quant, device_map="auto"
    )
    model.eval()
    return ("aya_gpu", tokenizer, model)


def _load_aya_cpu():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(repo_id=AYA_GGUF_REPO, filename=AYA_GGUF_FILE)
    llm = Llama(model_path=path, n_ctx=8192, n_threads=None, verbose=False)
    return ("aya_cpu", llm)


def _load_nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLLB_REPO, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_REPO).to(get_device())
    model.eval()
    return ("nllb", tokenizer, model)


def _load_translator():
    if has_cuda():
        try:
            return _load_aya_gpu()
        except Exception as exc:
            logger.warning("Aya GPU load failed (%s); trying GGUF", exc)
    try:
        return _load_aya_cpu()
    except Exception as exc:
        logger.warning("Aya GGUF load failed (%s); falling back to NLLB", exc)
    return _load_nllb()


# ── Generation ───────────────────────────────────────────────────────────────

def _build_prompt(regions: list[Region], glossary: Glossary) -> str:
    lines = []
    for i, reg in enumerate(regions, 1):
        tag = "[SFX] " if reg.is_sfx else ""
        lines.append(f"{i}. {tag}{reg.text_en}")
    return (
        "This is one page of manga/comic dialogue, in reading order."
        + glossary.as_prompt_block()
        + "\nTranslate each numbered line to Arabic:\n"
        + "\n".join(lines)
    )


def _generate_aya_gpu(bundle, user_prompt: str) -> str:
    import torch

    _, tokenizer, model = bundle
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def _generate_aya_cpu(bundle, user_prompt: str) -> str:
    _, llm = bundle
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return result["choices"][0]["message"]["content"]


def _parse_numbered(output: str, expected: int) -> list[str] | None:
    found: dict[int, str] = {}
    for line in output.splitlines():
        m = NUMBERED_LINE.match(line)
        if m:
            found[int(m.group(1))] = m.group(2).strip()
    if len(found) < expected:
        return None
    result = [found.get(i, "") for i in range(1, expected + 1)]
    if any(not is_arabic(t) for t in result):
        return None
    return result


def _translate_llm(bundle, regions: list[Region], glossary: Glossary) -> list[str] | None:
    prompt = _build_prompt(regions, glossary)
    generate = _generate_aya_gpu if bundle[0] == "aya_gpu" else _generate_aya_cpu
    for attempt in range(2):  # one retry on validation failure
        output = generate(bundle, prompt)
        parsed = _parse_numbered(output, len(regions))
        if parsed is not None:
            return parsed
        logger.warning("LLM output failed validation (attempt %d)", attempt + 1)
    return None


def _translate_nllb(bundle, regions: list[Region]) -> list[str]:
    import torch

    _, tokenizer, model = bundle
    results: list[str] = []
    ar_token = tokenizer.convert_tokens_to_ids("arb_Arab")
    for reg in regions:
        inputs = tokenizer(reg.text_en, return_tensors="pt").to(get_device())
        with torch.no_grad():
            out = model.generate(
                **inputs, forced_bos_token_id=ar_token, max_new_tokens=256
            )
        results.append(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
    return results


# ── Public API ───────────────────────────────────────────────────────────────

def translate_page(regions: list[Region], glossary: Glossary | None = None) -> list[Region]:
    """Translate all regions of a page with page-level context. Fills reg.text_ar."""
    if not regions:
        return regions
    glossary = glossary or Glossary()
    bundle = registry.swap_in("translator", _load_translator)

    translations: list[str] | None = None
    if bundle[0] in ("aya_gpu", "aya_cpu"):
        translations = _translate_llm(bundle, regions, glossary)
        if translations is None:
            logger.warning("LLM translation failed twice; falling back to NLLB")
            registry.unload("translator")
            bundle = registry.swap_in("translator", _load_nllb)

    if translations is None:
        translations = _translate_nllb(bundle, regions)

    for reg, ar in zip(regions, translations):
        reg.text_ar = ar.strip()
    return regions
