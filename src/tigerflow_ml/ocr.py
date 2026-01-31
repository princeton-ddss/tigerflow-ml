"""
Perform OCR on images using Hugging Face image-to-text models.

Loads an OCR model once in setup() and reuses it across all images.
Supports any Hugging Face model compatible with the image-to-text pipeline.

Usage:
    python -m tigerflow_ml.ocr \
        --input-dir ./images \
        --output-dir ./text \
        --input-ext .png \
        --output-ext .txt \
        --model microsoft/trocr-base-printed \
        --max-length 512 \
        --batch-size 1

For Slurm:
    python -m tigerflow_ml.ocr \
        --input-dir ./images \
        --output-dir ./text \
        --input-ext .png \
        --output-ext .txt \
        --model microsoft/trocr-base-printed \
        --account myaccount \
        --max-workers 4 \
        --cpus 4 \
        --memory 32GB \
        --time 04:00:00 \
        --gpus 1
"""

from pathlib import Path

import typer
from typing_extensions import Annotated

from tigerflow.tasks import SlurmTask
from tigerflow.utils import SetupContext


class OCR(SlurmTask):
    """Extract text from images using Hugging Face image-to-text models."""

    class Params:
        model: Annotated[
            str,
            typer.Option(
                help="Hugging Face model name (e.g., microsoft/trocr-base-printed)"
            ),
        ] = "microsoft/trocr-base-printed"

        max_length: Annotated[
            int,
            typer.Option(help="Maximum length of generated text"),
        ] = 512

        batch_size: Annotated[
            int,
            typer.Option(help="Batch size for processing"),
        ] = 1

        device: Annotated[
            str,
            typer.Option(help="Device to use (cuda, cpu, or auto)"),
        ] = "auto"

    @staticmethod
    def setup(context: SetupContext):
        from transformers import pipeline

        device_map = context.device
        if device_map == "auto":
            import torch

            device_map = 0 if torch.cuda.is_available() else -1

        context.ocr_pipeline = pipeline(
            "image-to-text",
            model=context.model,
            device=device_map,
            batch_size=context.batch_size,
        )
        context._max_length = context.max_length

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        from PIL import Image

        image = Image.open(input_file)

        # Convert to RGB if necessary (some models require RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        result = context.ocr_pipeline(image, max_new_tokens=context._max_length)

        # Extract generated text from result
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "")
        else:
            text = ""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    OCR.cli()
