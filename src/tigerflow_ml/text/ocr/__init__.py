from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.text.ocr.slurm import OCR

__all__ = ["OCR"]


def __getattr__(name: str):
    if name == "OCR":
        from tigerflow_ml.text.ocr.slurm import OCR

        return OCR
    raise AttributeError(f"module 'tigerflow_ml.text.ocr' has no attribute {name!r}")
