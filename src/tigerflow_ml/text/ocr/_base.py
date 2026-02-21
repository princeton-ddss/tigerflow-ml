"""
Perform OCR on images using Hugging Face vision-language models.

Supports TrOCR models and other VLMs compatible with image-text-to-text pipeline.
"""

from pathlib import Path
from typing import Annotated

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams


class _OCRBase:
    """Extract text from images using vision-language models."""

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

        prompt: Annotated[
            str,
            typer.Option(help="Prompt for VLM models (ignored for TrOCR)"),
        ] = "Extract all text from this image."

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from transformers import AutoConfig

        logger.info("Setting up OCR model...")
        logger.info("Model: {}", context.model)

        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check model type to determine loading strategy
        config = AutoConfig.from_pretrained(context.model)
        is_trocr = "trocr" in context.model.lower() or config.model_type == "vision-encoder-decoder"

        if is_trocr:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            context.processor = TrOCRProcessor.from_pretrained(
                context.model,
                revision=context.revision,
                cache_dir=context.cache_dir or None,
            )
            context.ocr_model = VisionEncoderDecoderModel.from_pretrained(
                context.model,
                revision=context.revision,
                cache_dir=context.cache_dir or None,
            )
            context.ocr_model.to(device)
            context.is_trocr = True
        else:
            from transformers import pipeline

            context.pipeline = pipeline(
                "image-text-to-text",
                model=context.model,
                revision=context.revision,
                cache_dir=context.cache_dir or None,
                device=device,
            )
            context.is_trocr = False

        context.ocr_device = device
        logger.info("OCR ready on device: {}", device)

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        import torch

        logger.info("Processing: {}", input_file)
        images = _load_images(input_file)
        logger.info("Loaded {} image(s)", len(images))

        pages = []
        for i, image in enumerate(images):
            if context.is_trocr:
                pixel_values = context.processor(
                    images=image, return_tensors="pt"
                ).pixel_values
                pixel_values = pixel_values.to(context.ocr_device)

                with torch.no_grad():
                    generated_ids = context.ocr_model.generate(
                        pixel_values,
                        max_length=context.max_length,
                    )

                text = context.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            else:
                result = context.pipeline(
                    image,
                    prompt=context.prompt,
                    max_new_tokens=context.max_length,
                )
                text = result[0].get("generated_text", "")

            pages.append(text)
            logger.info("Page {}: {} chars", i + 1, len(text))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\f".join(pages))

        logger.info("Done")


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
