"""
Perform OCR on images using Hugging Face image-to-text models.

Supports any Hugging Face model compatible with the image-to-text pipeline.
"""

from pathlib import Path
from typing import Annotated

import typer
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams


class _OCRBase:
    """Extract text from images using Hugging Face image-to-text models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "microsoft/trocr-base-printed"

        max_length: Annotated[
            int,
            typer.Option(help="Maximum length of generated text"),
        ] = 512

        batch_size: Annotated[
            int,
            typer.Option(help="Number of images to process in parallel on GPU"),
        ] = 4

    @staticmethod
    def setup(context: SetupContext):
        from transformers import pipeline

        device_map = context.device
        if device_map == "auto":
            import torch

            device_map = 0 if torch.cuda.is_available() else -1

        context.pipeline = pipeline(  # type: ignore[call-overload]
            "image-to-text",
            model=context.model,
            revision=context.revision,
            cache_dir=context.cache_dir or None,
            device=device_map,
            local_files_only=bool(context.cache_dir),
        )

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        images = _load_images(input_file)
        results = context.pipeline(
            images,
            max_new_tokens=context.max_length,
            batch_size=context.batch_size,
        )
        pages = [
            r[0].get("generated_text", "") if isinstance(r, list) and r else ""
            for r in results
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\f".join(pages))


def _load_images(path: Path) -> list:
    """Load images from a file. Supports image files and PDFs."""
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        import pymupdf

        images = []
        with pymupdf.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(image)
        return images

    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [image]
