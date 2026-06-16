"""Serialize a :class:`Transcription` to text, SRT, or JSON."""

from __future__ import annotations

from enum import Enum

from .transcriber import Transcription


class OutputFormat(str, Enum):
    """Output format for audio transcription."""

    TEXT = "text"
    SRT = "srt"
    JSON = "json"


def to_text(transcription: Transcription) -> str:
    """Plain transcript text."""
    return transcription.text.strip()


def to_json(transcription: Transcription) -> str:
    """JSON matching the speech-recognition-inference schema.

    Shape: ``{"language", "text", "chunks": [{"text", "timestamp"}]}``.
    """
    return transcription.model_dump_json(indent=2)


def to_srt(transcription: Transcription) -> str:
    """SubRip (.srt) subtitles built from the transcription's chunks.

    Empty chunks are skipped and the remaining ones are numbered
    sequentially (the index reflects emitted cues, not raw chunk position).
    """
    lines: list[str] = []
    index = 1
    for chunk in transcription.chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        start, end = chunk.timestamp
        lines.append(str(index))
        lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        lines.append(text)
        lines.append("")
        index += 1
    return "\n".join(lines)


def serialize(transcription: Transcription, output_format: OutputFormat) -> str:
    """Dispatch to the serializer for ``output_format``."""
    if output_format == OutputFormat.JSON:
        return to_json(transcription)
    if output_format == OutputFormat.SRT:
        return to_srt(transcription)
    return to_text(transcription)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to an SRT timestamp (``HH:MM:SS,mmm``)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
