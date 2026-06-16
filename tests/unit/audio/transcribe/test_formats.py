"""Unit tests for transcription output formats (no model downloads)."""

import json

from tigerflow_ml.audio.transcribe.formats import (
    OutputFormat,
    _format_timestamp,
    serialize,
    to_json,
    to_srt,
    to_text,
)
from tigerflow_ml.audio.transcribe.transcriber import Transcription


def _make(chunks, text=None, language="en"):
    text = text if text is not None else "".join(c[0] for c in chunks)
    return Transcription(
        language=language,
        text=text,
        chunks=[{"text": t, "timestamp": ts} for t, ts in chunks],
    )


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_fractional_seconds(self):
        assert _format_timestamp(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert _format_timestamp(65.25) == "00:01:05,250"

    def test_hours(self):
        assert _format_timestamp(3661.0) == "01:01:01,000"


def test_to_text_strips():
    t = _make([("  hello world  ", (0.0, 1.0))])
    assert to_text(t) == "hello world"


def test_to_json_matches_service_schema():
    t = _make([(" hi", (0.0, 1.5))], text=" hi", language="en")
    data = json.loads(to_json(t))
    assert set(data) == {"language", "text", "chunks"}
    assert data["language"] == "en"
    assert data["text"] == " hi"
    assert data["chunks"] == [{"text": " hi", "timestamp": [0.0, 1.5]}]


class TestSrt:
    def test_basic_cues(self):
        t = _make([(" one", (0.0, 2.0)), (" two", (2.0, 4.0))])
        srt = to_srt(t)
        assert srt.startswith("1\n00:00:00,000 --> 00:00:02,000\none")
        assert "2\n00:00:02,000 --> 00:00:04,000\ntwo" in srt

    def test_skips_empty_and_numbers_sequentially(self):
        # An empty chunk between two real ones must not create a gap in the
        # cue numbering (the #108 bug).
        t = _make(
            [
                (" first", (0.0, 1.0)),
                ("   ", (1.0, 2.0)),
                (" third", (2.0, 3.0)),
            ]
        )
        srt = to_srt(t)
        assert "1\n" in srt.split("\n\n")[0]
        # Second emitted cue is numbered 2, not 3.
        cues = [block for block in srt.split("\n\n") if block.strip()]
        assert cues[1].startswith("2\n")
        assert "third" in cues[1]
        assert len(cues) == 2

    def test_timestamp_formatting_hours(self):
        t = _make([(" x", (3661.5, 3662.25))])  # 1h 1m 1.5s
        srt = to_srt(t)
        assert "01:01:01,500 --> 01:01:02,250" in srt


def test_serialize_dispatch():
    t = _make([(" hi", (0.0, 1.0))])
    assert serialize(t, OutputFormat.TEXT) == to_text(t)
    assert serialize(t, OutputFormat.JSON) == to_json(t)
    assert serialize(t, OutputFormat.SRT) == to_srt(t)
