"""
Transcribe audio to text using Hugging Face Whisper models.

Audio is decoded in overlapping 30s windows with ``WhisperForConditional
Generation`` and stitched into a single transcription, matching the contract
of the ``princeton-ddss/speech-recognition-inference`` service. Output can be
plain text, SRT subtitles, or JSON.
"""

from pathlib import Path
from typing import Annotated

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

from .formats import OutputFormat, serialize
from .transcriber import load_whisper, transcribe_audio

__all__ = ["OutputFormat", "_TranscribeBase"]


class _TranscribeBase:
    """Transcribe audio to text using Hugging Face Whisper models."""

    class Params(HFParams):
        language: Annotated[
            str,
            typer.Option(
                help="Source language code (e.g. 'en', 'de', 'fr'). "
                "Leave empty to let the model detect it per file."
            ),
        ] = ""

        output_format: Annotated[
            OutputFormat,
            typer.Option(help="Output format: 'text', 'srt', or 'json'"),
        ] = OutputFormat.TEXT

        batch_size: Annotated[
            int,
            typer.Option(
                help="Number of 30s audio windows decoded per batch. "
                "Higher values use more GPU memory.",
                min=1,
            ),
        ] = 16

        overlap_s: Annotated[
            float,
            typer.Option(
                help="Overlap in seconds between consecutive 30s windows. "
                "Overlap is de-duplicated when stitching, recovering words "
                "that straddle a window boundary.",
                min=0.0,
                max=15.0,
            ),
        ] = 5.0

    @staticmethod
    def setup(context: SetupContext):
        whisper, processor, device = load_whisper(
            model=context.model,
            revision=context.revision,
            cache_dir=context.cache_dir,
            allow_fetch=context.allow_fetch,
            device=context.device,
            seed=context.seed,
        )
        context.whisper = whisper
        context.processor = processor
        context.device = device

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        logger.info(f"Transcribing: {input_file}")

        transcription = transcribe_audio(
            input_file,
            context.whisper,
            context.processor,
            context.device,
            language=context.language or None,
            batch_size=context.batch_size,
            overlap_s=context.overlap_s,
        )

        output_text = serialize(transcription, context.output_format)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        logger.info(f"Wrote {len(output_text)} characters to: {output_file}")
        logger.info("Done")
