"""Serialize a :class:`TranscriptionResult` to text, SRT, JSON, or raw.

The text/SRT/JSON formats merge the overlapping windows into one clean
transcript (loss-aversely; see :func:`~.transcriber.merge_overlapping`). The
``raw`` format does no merging: it emits every window's segments with overlap
annotations so a consumer (or an LLM) can reconcile them.
"""

from __future__ import annotations

import json
from enum import Enum

from .transcriber import Transcription, TranscriptionResult, merge_overlapping


class OutputFormat(str, Enum):
    """Output format for audio transcription."""

    TEXT = "text"
    SRT = "srt"
    JSON = "json"
    RAW = "raw"


def _merged(result: TranscriptionResult) -> Transcription:
    """Merge a result's windows into one transcription."""
    return merge_overlapping(
        [w.transcription for w in result.windows],
        language=result.language,
    )


def to_text(result: TranscriptionResult) -> str:
    """Plain transcript text (merged)."""
    return _merged(result).text.strip()


def to_json(result: TranscriptionResult) -> str:
    """Merged JSON matching the speech-recognition-inference schema.

    Shape: ``{"language", "text", "chunks": [{"text", "timestamp"}]}``.
    """
    return _merged(result).model_dump_json(indent=2)


def to_srt(result: TranscriptionResult) -> str:
    """SubRip (.srt) subtitles built from the merged transcription's chunks.

    Empty chunks are skipped and the remaining ones are numbered
    sequentially (the index reflects emitted cues, not raw chunk position).
    """
    lines: list[str] = []
    index = 1
    for chunk in _merged(result).chunks:
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


def to_raw(result: TranscriptionResult) -> str:
    """Un-merged JSON: every window's segments, with overlap annotations.

    Shape::

        {
          "language": "en",
          "overlap_s": 5.0,
          "segments": [
            {"text": ..., "timestamp": [s, e], "window": i, "overlap": bool},
            ...
          ]
        }

    A segment is marked ``overlap: true`` when its start falls inside the
    region shared with the next window (i.e. it has a redundant counterpart in
    that window). Consumers can drop ``overlap`` segments for a quick transcript
    or reconcile them for an exact one. Segments are listed window by window,
    in time order.
    """
    # The start of each window's shared region with the next window.
    next_offsets = [w.offset for w in result.windows[1:]] + [float("inf")]

    segments = []
    for window, next_start in zip(result.windows, next_offsets):
        for chunk in window.transcription.chunks:
            start, end = chunk.timestamp
            segments.append(
                {
                    "text": chunk.text,
                    "timestamp": [start, end],
                    "window": window.index,
                    "overlap": end > next_start,
                }
            )

    payload = {
        "language": result.language,
        "overlap_s": result.overlap_s,
        "segments": segments,
    }
    return json.dumps(payload, indent=2)


def serialize(result: TranscriptionResult, output_format: OutputFormat) -> str:
    """Dispatch to the serializer for ``output_format``."""
    if output_format == OutputFormat.JSON:
        return to_json(result)
    if output_format == OutputFormat.SRT:
        return to_srt(result)
    if output_format == OutputFormat.RAW:
        return to_raw(result)
    return to_text(result)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to an SRT timestamp (``HH:MM:SS,mmm``)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
