"""
Transcribe audio to text using Hugging Face Whisper models.

By default, audio is decoded in overlapping 30s windows with
``WhisperForConditionalGeneration`` (batched on the GPU) and stitched into a
single transcription, matching the contract of the
``princeton-ddss/speech-recognition-inference`` service. ``--windowing native``
instead uses Whisper's sequential long-form algorithm for the cleanest,
seam-free transcript at the cost of speed. The output format follows the output
file extension (``.srt``, ``.json``, else plain text); for ``.json``, ``--raw``
emits un-merged per-window segments instead of the merged transcript.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

from .formats import serialize
from .transcriber import load_whisper, transcribe_audio, transcribe_audio_native

__all__ = ["Windowing", "_TranscribeBase"]


class Windowing(str, Enum):
    """Decode strategy for long audio."""

    BATCHED = "batched"
    NATIVE = "native"


class _TranscribeBase:
    """Transcribe audio to text using Hugging Face Whisper models."""

    class Params(HFParams):
        language: Annotated[
            str,
            typer.Option(
                help="Source language code (e.g. 'en', 'de', 'fr'). "
                "Leave empty to let the model detect source language per file."
            ),
        ] = ""

        raw: Annotated[
            bool,
            typer.Option(
                help="For .json output only: emit un-merged per-window segments "
                "with overlap annotations (for exact reconciliation downstream) "
                "instead of the merged transcript. Ignored for other output "
                "extensions."
            ),
        ] = False

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
                "that straddle a window boundary. Ignored when --windowing is "
                "'native'.",
                min=0.0,
                max=15.0,
            ),
        ] = 5.0

        windowing: Annotated[
            Windowing,
            typer.Option(
                help="Decode strategy. 'batched' cuts audio into overlapping "
                "30s windows decoded in parallel (fast). 'native' uses "
                "Whisper's sequential long-form algorithm, which produces the "
                "cleanest transcript, but is much slower."
            ),
        ] = Windowing.BATCHED

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

        if context.windowing == Windowing.NATIVE:
            result = transcribe_audio_native(
                input_file,
                context.whisper,
                context.processor,
                context.device,
                language=context.language or None,
            )
        else:
            result = transcribe_audio(
                input_file,
                context.whisper,
                context.processor,
                context.device,
                language=context.language or None,
                batch_size=context.batch_size,
                overlap_s=context.overlap_s,
            )

        output_text = serialize(result, output_file.suffix, raw=context.raw)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        logger.info(f"Wrote {len(output_text)} characters to: {output_file}")
        logger.info("Done")
