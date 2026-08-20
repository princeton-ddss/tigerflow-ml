from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.image.ocr.slurm import OCR

__all__ = ["OCR"]


def __getattr__(name: str):
    if name == "OCR":
        from tigerflow_ml.image.ocr.slurm import OCR

        return OCR
    raise AttributeError(f"module 'tigerflow_ml.image.ocr' has no attribute {name!r}")
