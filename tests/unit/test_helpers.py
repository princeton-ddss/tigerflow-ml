"""Unit tests for pure helper functions (no model downloads needed)."""

from tigerflow_ml.audio.transcribe._base import _format_as_srt, _format_timestamp


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_fractional_seconds(self):
        assert _format_timestamp(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert _format_timestamp(65.25) == "00:01:05,250"

    def test_hours(self):
        assert _format_timestamp(3661.0) == "01:01:01,000"


class TestFormatAsSrt:
    def test_empty_chunks(self):
        result = {"text": "Hello world", "chunks": []}
        assert _format_as_srt(result) == "Hello world"

    def test_single_chunk(self):
        result = {
            "text": "Hello",
            "chunks": [{"text": "Hello", "timestamp": (0.0, 1.5)}],
        }
        srt = _format_as_srt(result)
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "Hello" in srt

    def test_multiple_chunks(self):
        result = {
            "text": "Hello world",
            "chunks": [
                {"text": "Hello", "timestamp": (0.0, 1.0)},
                {"text": "world", "timestamp": (1.0, 2.0)},
            ],
        }
        srt = _format_as_srt(result)
        lines = srt.strip().split("\n")
        # Two subtitle blocks separated by blank line
        assert "1" in lines
        assert "2" in lines
