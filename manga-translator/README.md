# Manga & Comics Translator — English → Arabic

Quality-first, fully open-source pipeline behind a Gradio UI. Runs on CPU or GPU (Colab-compatible).

```
Input → 1. Bubble/Text Detection → 2. OCR → 3. Cleaning
      → 4. Translation (EN→AR)  → 5. Inpainting → 6. Arabic Typesetting → Output
```

## Stack

| Stage | Model / Library |
|---|---|
| 1. Detection | `kitsumed/yolov8m_seg-speech-bubble` + `ogkalu/comic-text-and-bubble-detector`, NMS-merged |
| 2. OCR | PaddleOCR v4 (en) + `microsoft/trocr-large-printed` ensemble, Real-ESRGAN small-text upscaling |
| 3. Cleaning | symspellpy spell-correction, ALL-CAPS fix, SFX tagging |
| 4. Translation | Aya-Expanse-8B — 4-bit (GPU) or GGUF Q4_K_M via llama-cpp (CPU); NLLB-200-1.3B automatic fallback |
| 5. Inpainting | LaMa (`iopaint`) for text over art; flood-fill for solid bubbles |
| 6. Typesetting | Pillow + libraqm (or arabic-reshaper + python-bidi), bundled Noto Naskh Arabic + Cairo |

## Quick start (local)

```bash
cd manga-translator
# Optional but recommended for best Arabic rendering:
sudo apt-get install libraqm0          # Linux
pip install -r requirements.txt
python app.py                          # opens the Gradio UI
```

## Quick start (Google Colab)

Open `colab_manga_translator.ipynb`, choose a **T4 GPU** runtime, run the single setup cell, then the launch cell. A public `share=True` link appears.

## Usage

1. Upload pages (PNG/JPG/WEBP) or a whole chapter as CBZ/ZIP.
2. Optionally pin character names in the **Glossary** table (English → Arabic) for consistency.
3. Click **Analyze & Translate** — detection, OCR, and page-context translation run.
4. **Review table**: correct any Arabic line before typesetting (the biggest quality lever).
5. Click **Apply Edits & Render** — text removal + RTL typesetting produce final pages.
6. Download individual PNGs or the repacked CBZ.

## Project structure

```
manga-translator/
├── app.py                  # Gradio interface
├── core/
│   ├── detector.py         # Stage 1 — YOLOv8-seg ensemble + NMS
│   ├── ocr.py              # Stage 2 — PaddleOCR + TrOCR ensemble
│   ├── cleaner.py          # Stage 3 — normalization, SFX tagging, spellfix
│   ├── translator.py       # Stage 4 — Aya LLM w/ page context + glossary
│   ├── inpainter.py        # Stage 5 — LaMa / flood-fill two-tier removal
│   ├── typesetter.py       # Stage 6 — RTL binary-search typesetting
│   └── pipeline.py         # Orchestrator (analyze / review / render phases)
├── utils/
│   ├── device.py           # CPU/GPU auto-detection + model registry
│   ├── arabic_text.py      # raqm detection, reshaper+bidi fallback, RTL wrap
│   └── image_ops.py        # crops, masks, NMS, color sampling
├── fonts/                  # Noto Naskh Arabic (body), Cairo (bold/SFX)
├── colab_manga_translator.ipynb
└── requirements.txt
```

## Memory notes (Colab)

- Models are lazy-loaded; only one heavy model (translator / inpainter / upscaler) stays in RAM/VRAM at a time.
- GPU: fp16 for vision models, 4-bit NF4 for the LLM (~6 GB VRAM — fits a T4).
- CPU: GGUF Q4_K_M via llama-cpp-python.
