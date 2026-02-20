"""
Translate text documents using Hugging Face translation models.

Supports any Hugging Face model compatible with the translation pipeline.
"""

import logging
import re
from pathlib import Path
from typing import Annotated

import typer
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

logger = logging.getLogger(__name__)


class _TranslateBase:
    """Translate documents using Hugging Face translation models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "Helsinki-NLP/opus-mt-en-de"

        max_length: Annotated[
            int,
            typer.Option(help="Maximum length of generated translation"),
        ] = 512

        batch_size: Annotated[
            int,
            typer.Option(help="Number of chunks to translate in parallel on GPU"),
        ] = 4

        encoding: Annotated[
            str,
            typer.Option(
                help="Input file encoding (non-UTF-8 may cause "
                "lossy tokenization)"
            ),
        ] = "utf-8-sig"

    @staticmethod
    def setup(context: SetupContext):
        from transformers import pipeline

        device_map = context.device
        if device_map == "auto":
            import torch

            device_map = 0 if torch.cuda.is_available() else -1

        context.pipeline = pipeline(  # type: ignore
            "translation",
            model=context.model,
            revision=context.revision,
            cache_dir=context.cache_dir or None,
            device=device_map,
            local_files_only=bool(context.cache_dir),
        )
        context.tokenizer = context.pipeline.tokenizer
        context.max_input_tokens = context.tokenizer.model_max_length

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        with open(input_file, encoding=context.encoding) as f:
            text = f.read()

        chunks = _chunk_text(text, context.tokenizer, context.max_input_tokens)
        results = context.pipeline(
            chunks,
            max_length=context.max_length,
            batch_size=context.batch_size,
        )

        translated_parts = []
        for chunk, result in zip(chunks, results):
            output = result["translation_text"]
            n_tokens = len(
                context.tokenizer.encode(output, add_special_tokens=False)
            )
            if n_tokens >= context.max_length:
                logger.warning(
                    "Translation may be truncated "
                    "(hit max_length=%d) for chunk: %.50s...",
                    context.max_length,
                    chunk,
                )
            translated_parts.append(output)

        translated = " ".join(translated_parts)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(translated)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s]


def _chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Pack sentences into chunks that fit within the token limit."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))

        if token_count > max_tokens:
            # Flush current chunk
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            # Force-split the long sentence at token boundaries
            chunks.extend(_split_by_tokens(sentence, tokenizer, max_tokens))
            continue

        if current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += token_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Last resort: split text at token boundaries.

    Note: the encode/decode round-trip may not preserve the original
    text exactly (e.g., unknown characters, whitespace normalization,
    Unicode normalization).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks
