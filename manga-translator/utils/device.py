"""
Device detection and memory management.

Rules (Colab-safe):
- GPU path : fp16 for detection / OCR / inpainting, 4-bit for the LLM.
- CPU path : fp32 for small models, GGUF Q4_K_M via llama-cpp for translation.
- Only ONE heavy model resident in memory at a time.
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger("manga_translator.device")

_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch  # heavy import, do it lazily

        _TORCH = torch
    return _TORCH


def has_cuda() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def get_device() -> str:
    """'cuda' or 'cpu'."""
    return "cuda" if has_cuda() else "cpu"


def get_dtype():
    """fp16 on GPU, fp32 on CPU."""
    torch = _torch()
    return torch.float16 if has_cuda() else torch.float32


def gpu_vram_gb() -> float:
    if not has_cuda():
        return 0.0
    torch = _torch()
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def free_memory() -> None:
    """Aggressively release memory between pipeline stages (Colab-critical)."""
    gc.collect()
    if has_cuda():
        torch = _torch()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class ModelRegistry:
    """
    Keeps at most one *heavy* model loaded at a time.

    Usage:
        registry.swap_in("translator", loader_fn)  # unloads everything else
    """

    HEAVY = {"translator", "inpainter", "upscaler"}

    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def get(self, name: str):
        return self._models.get(name)

    def swap_in(self, name: str, loader):
        if name in self._models:
            return self._models[name]
        if name in self.HEAVY:
            # evict other heavy models first
            for other in list(self._models):
                if other in self.HEAVY and other != name:
                    self.unload(other)
        logger.info("Loading model: %s (device=%s)", name, get_device())
        model = loader()
        self._models[name] = model
        return model

    def unload(self, name: str) -> None:
        model = self._models.pop(name, None)
        if model is not None:
            logger.info("Unloading model: %s", name)
            del model
            free_memory()

    def unload_all(self) -> None:
        for name in list(self._models):
            self.unload(name)


registry = ModelRegistry()
