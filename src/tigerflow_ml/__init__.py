# ML tasks for TigerFlow.
#
# Task classes are exposed lazily so that `import tigerflow_ml` (or importing
# any submodule) does not eagerly load every task's heavy dependencies
# (torch, transformers, vllm). Each name resolves on first attribute access.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.audio.transcribe.slurm import Transcribe
    from tigerflow_ml.image.detect.slurm import Detect
    from tigerflow_ml.text.chat.slurm import Chat
    from tigerflow_ml.text.ocr.slurm import OCR
    from tigerflow_ml.text.translate.slurm import Translate

_LAZY_TASKS = {
    "OCR": "tigerflow_ml.text.ocr.slurm",
    "Translate": "tigerflow_ml.text.translate.slurm",
    "Transcribe": "tigerflow_ml.audio.transcribe.slurm",
    "Detect": "tigerflow_ml.image.detect.slurm",
    "Chat": "tigerflow_ml.text.chat.slurm",
}

__all__ = ["OCR", "Translate", "Transcribe", "Detect", "Chat"]


def __getattr__(name: str):
    module_path = _LAZY_TASKS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'tigerflow_ml' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
