from tigerflow.tasks import SlurmTask

from tigerflow_ml.text.ocr._base import _OCRBase


class OCR(_OCRBase, SlurmTask):
    """OCR task for Slurm execution."""


if __name__ == "__main__":
    OCR.cli()
