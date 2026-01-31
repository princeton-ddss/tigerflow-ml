"""
Translate text documents using Hugging Face text-to-text models.

Loads a translation model once in setup() and reuses it across all documents.
Supports any Hugging Face model compatible with the text2text-generation pipeline.

Usage:
    python -m tigerflow_ml.translate \
        --input-dir ./documents \
        --output-dir ./translated \
        --input-ext .txt \
        --output-ext .txt \
        --model Helsinki-NLP/opus-mt-en-es \
        --max-length 512 \
        --batch-size 1

For Slurm:
    python -m tigerflow_ml.translate \
        --input-dir ./documents \
        --output-dir ./translated \
        --input-ext .txt \
        --output-ext .txt \
        --model Helsinki-NLP/opus-mt-en-es \
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


class Translate(SlurmTask):
    """Translate documents using Hugging Face text-to-text models."""

    class Params:
        model: Annotated[
            str,
            typer.Option(
                help="Hugging Face model name (e.g., Helsinki-NLP/opus-mt-en-es)"
            ),
        ] = "Helsinki-NLP/opus-mt-en-de"

        max_length: Annotated[
            int,
            typer.Option(help="Maximum length of generated translation"),
        ] = 512

        batch_size: Annotated[
            int,
            typer.Option(help="Batch size for processing long documents"),
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

        context.translator = pipeline(
            "translation",
            model=context.model,
            device=device_map,
            batch_size=context.batch_size,
        )
        context._max_length = context.max_length

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Handle long documents by splitting into chunks if needed
        # Most translation models have a max input length
        max_chunk_length = 1000  # Characters per chunk (conservative estimate)

        if len(text) <= max_chunk_length:
            result = context.translator(text, max_length=context._max_length)
            translated = result[0]["translation_text"]
        else:
            # Split by paragraphs to preserve structure
            paragraphs = text.split("\n\n")
            translated_paragraphs = []

            for para in paragraphs:
                if para.strip():
                    result = context.translator(
                        para.strip(), max_length=context._max_length
                    )
                    translated_paragraphs.append(result[0]["translation_text"])
                else:
                    translated_paragraphs.append("")

            translated = "\n\n".join(translated_paragraphs)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(translated)


if __name__ == "__main__":
    Translate.cli()
