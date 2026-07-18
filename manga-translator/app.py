"""
Gradio UI for the English -> Arabic manga/comic translator.

Flow:
  1. Upload images (PNG/JPG/WEBP) or CBZ/ZIP archives.
  2. "Analyze & Translate" — runs detection, OCR, cleaning, translation.
  3. Human review table [# , English, Arabic] — edit any line.
  4. "Apply Edits & Render" — inpaints and typesets the final pages.
  5. Download individual pages or a re-packed CBZ.

Colab: `demo.launch(share=True)`. Models are lazy-loaded on first use.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from core.pipeline import PageJob, analyze_page, apply_review, render_page, review_rows
from core.translator import Glossary
from utils.device import get_device, has_cuda

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("manga_translator.app")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ARCHIVE_EXTS = {".cbz", ".zip"}


# ── File handling ────────────────────────────────────────────────────────────

def _expand_inputs(files: list) -> list[tuple[str, np.ndarray]]:
    """Expand uploaded files (images + CBZ/ZIP archives) into (name, BGR) pairs,
    keeping archive page order."""
    pages: list[tuple[str, np.ndarray]] = []
    for f in files or []:
        path = Path(f.name if hasattr(f, "name") else f)
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            img = cv2.imread(str(path))
            if img is not None:
                pages.append((path.stem, img))
        elif ext in ARCHIVE_EXTS:
            with zipfile.ZipFile(path) as zf:
                names = sorted(
                    n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS
                )
                for n in names:
                    data = np.frombuffer(zf.read(n), dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is not None:
                        pages.append((f"{path.stem}_{Path(n).stem}", img))
    return pages


def _save_outputs(jobs: list[PageJob]) -> tuple[list[str], str | None]:
    """Save rendered pages as PNGs; also pack a CBZ when there are 2+ pages."""
    out_dir = Path(tempfile.mkdtemp(prefix="manga_ar_"))
    paths: list[str] = []
    for i, job in enumerate(jobs):
        if job.output_bgr is None:
            continue
        p = out_dir / f"{i:03d}_{job.name}_ar.png"
        cv2.imwrite(str(p), job.output_bgr)
        paths.append(str(p))
    cbz_path = None
    if len(paths) > 1:
        cbz_path = str(out_dir / "translated_ar.cbz")
        with zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_STORED) as zf:
            for p in paths:
                zf.write(p, Path(p).name)
    return paths, cbz_path


# ── App state ────────────────────────────────────────────────────────────────

class SessionState:
    def __init__(self) -> None:
        self.jobs: list[PageJob] = []
        self.glossary = Glossary()


# ── Handlers ─────────────────────────────────────────────────────────────────

def handle_analyze(files, quality_mode, glossary_rows, state: SessionState, progress=gr.Progress()):
    if not files:
        raise gr.Error("Upload at least one image or CBZ/ZIP file first.")
    quality_max = quality_mode == "Maximum Quality"
    state.glossary.from_rows(
        glossary_rows.values.tolist() if hasattr(glossary_rows, "values") else glossary_rows
    )
    pages = _expand_inputs(files)
    if not pages:
        raise gr.Error("No readable images found in the upload.")

    state.jobs = []
    all_rows: list[list] = []
    for i, (name, img) in enumerate(pages):
        progress((i, len(pages)), desc=f"Translating page {i + 1}/{len(pages)}: {name}")
        job = analyze_page(img, state.glossary, quality_max=quality_max, name=name)
        state.jobs.append(job)
        for row in review_rows(job):
            all_rows.append([f"{i + 1}", *row])  # [page, #, EN, AR]

    detected = sum(len(j.regions or []) for j in state.jobs)
    msg = (
        f"Analyzed {len(state.jobs)} page(s), found {detected} text region(s). "
        "Review and correct the Arabic below, then click 'Apply Edits & Render'."
    )
    return (
        gr.update(value=all_rows, visible=True),
        gr.update(visible=True),
        msg,
        state,
    )


def handle_render(review_table, quality_mode, state: SessionState, progress=gr.Progress()):
    if not state.jobs:
        raise gr.Error("Run 'Analyze & Translate' first.")
    quality_max = quality_mode == "Maximum Quality"

    rows = review_table.values.tolist() if hasattr(review_table, "values") else (review_table or [])
    # Split rows back per page: rows are [page, #, EN, AR].
    per_page: dict[int, list[list]] = {}
    for row in rows:
        try:
            page_idx = int(row[0]) - 1
        except (ValueError, TypeError):
            continue
        per_page.setdefault(page_idx, []).append(row[1:])

    gallery: list[tuple[str, str]] = []
    for i, job in enumerate(state.jobs):
        progress((i, len(state.jobs)), desc=f"Rendering page {i + 1}/{len(state.jobs)}")
        apply_review(job, per_page.get(i, []))
        render_page(job, quality_max=quality_max)

    paths, cbz = _save_outputs(state.jobs)
    for path, job in zip(paths, state.jobs):
        gallery.append((path, job.name))

    downloads = paths + ([cbz] if cbz else [])
    return (
        gallery,
        gr.update(value=downloads, visible=True),
        f"Done — rendered {len(paths)} page(s)." + (" CBZ included." if cbz else ""),
        state,
    )


def handle_compare(evt: gr.SelectData, state: SessionState):
    """Show original vs translated for the clicked gallery item."""
    idx = evt.index if isinstance(evt.index, int) else evt.index[0]
    if idx is None or idx >= len(state.jobs):
        return None, None
    job = state.jobs[idx]
    orig = cv2.cvtColor(job.original_bgr, cv2.COLOR_BGR2RGB)
    out = cv2.cvtColor(job.output_bgr, cv2.COLOR_BGR2RGB) if job.output_bgr is not None else None
    return orig, out


# ── UI ───────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    device_label = f"Running on: {get_device().upper()}" + (
        " (4-bit LLM)" if has_cuda() else " (GGUF Q4_K_M LLM)"
    )

    with gr.Blocks(title="Manga & Comics — English to Arabic Translator") as demo:
        state = gr.State(SessionState())

        gr.Markdown(
            "# Manga & Comics Translator — English → Arabic\n"
            f"Quality-first pipeline: YOLOv8-seg → PaddleOCR+TrOCR → Aya LLM → "
            f"LaMa → RTL typesetting. {device_label}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="Pages (PNG / JPG / WEBP / CBZ / ZIP)",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".cbz", ".zip"],
                )
                quality_mode = gr.Radio(
                    ["Maximum Quality", "Fast"],
                    value="Maximum Quality",
                    label="Quality mode",
                    info="Maximum Quality = OCR ensemble + Real-ESRGAN + LaMa everywhere.",
                )
                glossary_table = gr.Dataframe(
                    headers=["English", "Arabic"],
                    datatype=["str", "str"],
                    row_count=(1, "dynamic"),
                    col_count=(2, "fixed"),
                    label="Glossary — pin fixed Arabic renderings for names/terms",
                    interactive=True,
                )
                analyze_btn = gr.Button("1 — Analyze & Translate", variant="primary")
                status = gr.Markdown("")

            with gr.Column(scale=2):
                review_table = gr.Dataframe(
                    headers=["Page", "#", "English", "Arabic"],
                    datatype=["str", "number", "str", "str"],
                    label="Review & edit translations before typesetting",
                    interactive=True,
                    visible=False,
                    wrap=True,
                )
                render_btn = gr.Button(
                    "2 — Apply Edits & Render", variant="primary", visible=False
                )
                gallery = gr.Gallery(label="Translated pages", columns=3, height="auto")
                with gr.Row():
                    compare_orig = gr.Image(label="Original", interactive=False)
                    compare_out = gr.Image(label="Translated", interactive=False)
                downloads = gr.File(
                    label="Download (PNG pages + CBZ)", file_count="multiple", visible=False
                )

        analyze_btn.click(
            handle_analyze,
            inputs=[files, quality_mode, glossary_table, state],
            outputs=[review_table, render_btn, status, state],
        )
        render_btn.click(
            handle_render,
            inputs=[review_table, quality_mode, state],
            outputs=[gallery, downloads, status, state],
        )
        gallery.select(handle_compare, inputs=[state], outputs=[compare_orig, compare_out])

    return demo


if __name__ == "__main__":
    demo = build_app()
    # share=True makes it Colab-compatible out of the box.
    demo.launch(share=True, server_name="0.0.0.0")
