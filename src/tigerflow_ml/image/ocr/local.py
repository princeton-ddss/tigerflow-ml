from tigerflow.tasks import LocalTask

from tigerflow_ml.image.ocr._base import _OCRBase


class OCR(_OCRBase, LocalTask):
    """OCR task for local execution."""


if __name__ == "__main__":
    OCR.cli()
