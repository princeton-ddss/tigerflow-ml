"""Unit tests for transcription output formats (no model downloads)."""

import json

from tigerflow_ml.audio.transcribe.formats import (
    OutputFormat,
    _format_timestamp,
    serialize,
    to_json,
    to_raw,
    to_srt,
    to_text,
)
from tigerflow_ml.audio.transcribe.transcriber import (
    Transcription,
    TranscriptionResult,
    Window,
)


def _window(chunks, index=0, offset=0.0, language="en"):
    """Build one Window from (text, (start, end)) chunk tuples."""
    return Window(
        index=index,
        offset=offset,
        transcription=Transcription(
            language=language,
            text="".join(c[0] for c in chunks),
            chunks=[{"text": t, "timestamp": ts} for t, ts in chunks],
        ),
    )


def _make(chunks, language="en", overlap_s=5.0):
    """A single-window result (merge is a passthrough; text is derived)."""
    return TranscriptionResult(
        language=language,
        windows=[_window(chunks, language=language)],
        overlap_s=overlap_s,
    )


def _result(windows, language="en", overlap_s=5.0):
    """A multi-window result from a list of Window objects."""
    return TranscriptionResult(language=language, windows=windows, overlap_s=overlap_s)


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
    t = _make([(" hi", (0.0, 1.5))], language="en")
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


class TestRaw:
    def test_schema_and_window_tags(self):
        result = _result(
            [
                _window([(" a", (0.0, 2.0))], index=0, offset=0.0),
                _window([(" b", (25.0, 27.0))], index=1, offset=25.0),
            ]
        )
        data = json.loads(to_raw(result))
        assert set(data) == {"language", "overlap_s", "segments"}
        assert data["language"] == "en"
        assert [s["window"] for s in data["segments"]] == [0, 1]
        assert data["segments"][0]["text"] == " a"
        assert data["segments"][0]["timestamp"] == [0.0, 2.0]

    def test_overlap_flag(self):
        # window 0 spans [0,30], window 1 starts at 25. A window-0 chunk ending
        # past 25 is in the shared region -> overlap True; one ending before is
        # not. The last window never overlaps a following one.
        result = _result(
            [
                _window(
                    [(" early", (10.0, 12.0)), (" late", (26.0, 28.0))],
                    index=0,
                    offset=0.0,
                ),
                _window([(" next", (30.0, 32.0))], index=1, offset=25.0),
            ]
        )
        data = json.loads(to_raw(result))
        flags = {s["text"]: s["overlap"] for s in data["segments"]}
        assert flags[" early"] is False
        assert flags[" late"] is True
        assert flags[" next"] is False  # last window

    def test_raw_keeps_duplicates_unmerged(self):
        # Both windows decode the same boundary word; raw keeps both copies.
        result = _result(
            [
                _window([(" word", (28.0, 29.5))], index=0, offset=0.0),
                _window([(" word", (28.0, 29.5))], index=1, offset=25.0),
            ]
        )
        data = json.loads(to_raw(result))
        assert [s["text"] for s in data["segments"]] == [" word", " word"]


def test_serialize_dispatch():
    t = _make([(" hi", (0.0, 1.0))])
    assert serialize(t, OutputFormat.TEXT) == to_text(t)
    assert serialize(t, OutputFormat.JSON) == to_json(t)
    assert serialize(t, OutputFormat.SRT) == to_srt(t)
    assert serialize(t, OutputFormat.RAW) == to_raw(t)
