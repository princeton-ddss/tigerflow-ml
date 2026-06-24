"""Serialize a :class:`TranscriptionResult` to text, SRT, or JSON.

The output format is chosen from the output file's extension (``.srt`` ->
subtitles, ``.json`` -> JSON, anything else -> plain text). The text/SRT/JSON
formats merge the overlapping windows into one clean transcript (loss-aversely;
see :func:`~.transcriber.merge_overlapping`). For ``.json`` output, the ``raw``
flag instead emits every window's segments un-merged, with overlap annotations,
so a consumer (or an LLM) can reconcile them.
"""

from __future__ import annotations

import json

from tigerflow.logconfig import logger

from .transcriber import Transcription, TranscriptionResult, merge_overlapping


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

    A segment is marked ``overlap: true`` when it extends into the region
    shared with the next window (its end is past where the next window begins),
    i.e. it likely has a redundant counterpart in that window. Consumers can
    drop ``overlap`` segments for a quick transcript or reconcile them for an
    exact one. Segments are listed window by window, in time order.
    """
    # The start time of the next window, against which a segment counts as
    # overlapping if it extends past it. The last window has no successor.
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


def serialize(result: TranscriptionResult, suffix: str, raw: bool = False) -> str:
    """Serialize a result, picking the format from the output file ``suffix``.

    ``.srt`` -> subtitles, ``.json`` -> JSON, anything else -> plain text. For
    ``.json`` output, ``raw`` emits un-merged per-window segments instead of the
    merged transcript; ``raw`` is ignored (with a warning) for other suffixes.

    Args:
        result: The transcription result.
        suffix: The output file's extension, e.g. ``".json"`` (case-insensitive).
        raw: Emit un-merged segments; only meaningful for ``.json`` output.

    Returns:
        The serialized output text.
    """
    suffix = suffix.lower()
    if suffix == ".json":
        return to_raw(result) if raw else to_json(result)
    if raw:
        logger.warning(
            f"--raw only applies to .json output; ignoring for '{suffix}' output"
        )
    if suffix == ".srt":
        return to_srt(result)
    return to_text(result)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to an SRT timestamp (``HH:MM:SS,mmm``)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
