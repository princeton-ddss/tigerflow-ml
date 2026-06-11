"""
Perform OCR on images using Hugging Face image-text-to-text models.

Supports VLMs compatible with the image-text-to-text pipeline.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

_DEFAULT_PROMPT = "Extract all text from this image."


class OutputFormat(str, Enum):
    """Output format for OCR results."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


class _OCRBase:
    """Extract text from images using image-text-to-text models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "stepfun-ai/GOT-OCR-2.0-hf"

        max_length: Annotated[
            int,
            typer.Option(help="Maximum number of tokens to generate per image"),
        ] = 4096

        output_format: Annotated[
            OutputFormat,
            typer.Option(help="Output format: 'text', 'markdown', or 'json'"),
        ] = OutputFormat.TEXT

        prompt: Annotated[
            str,
            typer.Option(help="Prompt for image-text-to-text models"),
        ] = _DEFAULT_PROMPT

        temperature: Annotated[
            float,
            typer.Option(
                help="The model temperature. "
                "Lower values make models more deterministic",
                min=0.0,
                max=2.0,
            ),
        ] = 0.0

        seed: Annotated[
            int, typer.Option(help="The seed to set for more reproducible behavior")
        ] = 42

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from transformers import pipeline, set_seed

        set_seed(context.seed)

        logger.info("Setting up OCR model...")
        logger.info("Model: {}", context.model)

        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            context.pipeline = pipeline(
                "image-text-to-text",
                model=context.model,
                revision=context.revision,
                device=device,
                local_files_only=not context.allow_fetch,
                model_kwargs={"cache_dir": context.cache_dir or None},
            )
        except OSError as e:
            if not context.allow_fetch:
                raise RuntimeError(
                    f"'{context.model}' not found in cache ({context.cache_dir}). "
                    "Run with --allow_fetch or download manually."
                ) from e
            raise
        logger.info("OCR ready on device: {}", device)

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        logger.info("Processing: {}", input_file)
        images = _load_images(input_file)
        logger.info("Loaded {} image(s)", len(images))

        # Determine effective prompt
        prompt = context.prompt
        if context.output_format == OutputFormat.MARKDOWN and prompt == _DEFAULT_PROMPT:
            prompt = "OCR with format"

        pages = []
        for i, image in enumerate(images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            gen_kwargs: dict = {"max_new_tokens": context.max_length}
            if context.temperature > 0:
                gen_kwargs["temperature"] = context.temperature
                gen_kwargs["do_sample"] = True
            result = context.pipeline(text=messages, **gen_kwargs)
            text = result[0]["generated_text"][-1]["content"]
            pages.append(text)
            logger.info("Page {}: {} chars", i + 1, len(text))

        if context.output_format == OutputFormat.JSON:
            output_text = json.dumps(
                {"pages": [{"page": i + 1, "text": p} for i, p in enumerate(pages)]},
                indent=2,
                ensure_ascii=False,
            )
        else:
            output_text = "\f".join(pages)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

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
