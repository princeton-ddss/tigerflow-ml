"""
Transcribe audio to text using Hugging Face Whisper models.

Supports transcription of audio files to plain text, SRT subtitles, or JSON.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams


class OutputFormat(str, Enum):
    """Output format for audio transcription."""

    TEXT = "text"
    SRT = "srt"
    JSON = "json"


class _TranscribeBase:
    """Transcribe audio to text using Hugging Face Whisper models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "openai/whisper-large-v3"

        language: Annotated[
            str,
            typer.Option(
                help="Source language code (e.g., 'en', 'de', 'fr'). "
                "Leave empty for auto-detection."
            ),
        ] = ""

        output_format: Annotated[
            OutputFormat,
            typer.Option(help="Output format: 'text', 'srt', or 'json'"),
        ] = OutputFormat.TEXT

        batch_size: Annotated[
            int,
            typer.Option(
                help="Batch size for processing audio chunks. "
                "Higher values use more GPU memory."
            ),
        ] = 16

        chunk_length_s: Annotated[
            float,
            typer.Option(
                help="Length of audio chunks in seconds for batched processing"
            ),
        ] = 30.0

        stride_length_s: Annotated[
            float,
            typer.Option(
                help="Overlap between chunks in seconds to avoid cutting words"
            ),
        ] = 5.0

        return_timestamps: Annotated[
            bool,
            typer.Option(
                help="Return word/segment timestamps (required for SRT output)"
            ),
        ] = False

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from transformers import AutoFeatureExtractor, pipeline

        logger.info("Setting up transcription pipeline...")
        logger.info("Model: {}", context.model)

        device_map = context.device
        if device_map == "auto":
            device_map = 0 if torch.cuda.is_available() else -1

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info("Device: {}, dtype: {}", device_map, torch_dtype)

        cache_dir = context.cache_dir or None

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            context.model,
            revision=context.revision,
            cache_dir=cache_dir,
        )

        context.pipeline = pipeline(
            "automatic-speech-recognition",
            model=context.model,
            revision=context.revision,
            device=device_map,
            torch_dtype=torch_dtype,
            model_kwargs={"cache_dir": cache_dir},
            feature_extractor=feature_extractor,
        )
        logger.info("Pipeline ready")

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        logger.info("Transcribing: {}", input_file)

        return_timestamps = context.return_timestamps
        if context.output_format in (OutputFormat.SRT, OutputFormat.JSON):
            return_timestamps = True

        generate_kwargs = {}
        if context.language:
            generate_kwargs["language"] = context.language

        logger.info(
            "Processing with batch_size={}, chunk_length_s={}",
            context.batch_size,
            context.chunk_length_s,
        )

        pipeline_kwargs = {
            "chunk_length_s": context.chunk_length_s,
            "stride_length_s": (
                context.stride_length_s,
                context.stride_length_s,
            ),
            "batch_size": context.batch_size,
            "return_timestamps": return_timestamps,
            "ignore_warning": True,
        }
        if generate_kwargs:
            pipeline_kwargs["generate_kwargs"] = generate_kwargs

        result = context.pipeline(str(input_file), **pipeline_kwargs)

        if context.output_format == OutputFormat.JSON:
            output_text = json.dumps(result, indent=2)
        elif context.output_format == OutputFormat.SRT:
            output_text = _format_as_srt(result)
        else:
            output_text = result["text"].strip()

        logger.info("Transcription complete, writing to: {}", output_file)
        logger.info("Output length: {} characters", len(output_text))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        logger.info("Done")


def _format_as_srt(result: dict) -> str:
    """Convert Whisper output with timestamps to SRT subtitle format."""
    chunks = result.get("chunks", [])
    if not chunks:
        return result.get("text", "").strip()

    srt_lines = []
    for i, chunk in enumerate(chunks, start=1):
        start_time = chunk.get("timestamp", (0, 0))[0] or 0
        end_time = chunk.get("timestamp", (0, 0))[1] or start_time + 1
        text = chunk.get("text", "").strip()

        if not text:
            continue

        srt_lines.append(str(i))
        srt_lines.append(
            f"{_format_timestamp(start_time)} --> {_format_timestamp(end_time)}"
        )
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
