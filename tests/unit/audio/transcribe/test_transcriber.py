"""Unit tests for the Whisper transcription engine (no model downloads)."""

from pathlib import Path

import numpy as np

from tigerflow_ml.audio.transcribe.transcriber import (
    SAMPLING_RATE,
    WINDOW_S,
    BatchIterator,
    Transcription,
    _flatten_sequences,
    load_audio,
    merge_overlapping,
)

_FIXTURE = (
    Path(__file__).parents[3] / "integration" / "fixtures" / "transcribe" / "sample.wav"
)


class TestLoadAudio:
    def test_returns_mono_float32_at_16khz(self):
        array = load_audio(_FIXTURE)
        assert array.dtype == np.float32
        assert array.ndim == 1
        # ~2.5s fixture at 16kHz.
        assert abs(len(array) / SAMPLING_RATE - 2.5) < 0.2


class TestFlattenSequences:
    def test_2d_list(self):
        assert _flatten_sequences([[1, 2, 3], [4, 5]]) == [[1, 2, 3], [4, 5]]

    def test_3d_concatenates_segments_per_window(self):
        # (batch=2, segments, seq): each window's segments are concatenated.
        ids = [[[1, 2], [3, 4]], [[5, 6, 7]]]
        assert _flatten_sequences(ids) == [[1, 2, 3, 4], [5, 6, 7]]

    def test_tensor_like_via_tolist(self):
        class FakeTensor:
            def tolist(self):
                return [[9, 8, 7]]

        assert _flatten_sequences(FakeTensor()) == [[9, 8, 7]]


class TestFromString:
    def test_parses_timestamp_segments(self):
        s = "<|0.00|> Hello world<|2.50|><|2.50|> goodbye<|4.00|>"
        t = Transcription.from_string(s, language="en")
        assert t.language == "en"
        assert t.text == " Hello world goodbye"
        assert len(t.chunks) == 2
        assert t.chunks[0].timestamp == (0.0, 2.5)
        assert t.chunks[1].timestamp == (2.5, 4.0)

    def test_no_timestamps_falls_back_to_single_chunk(self):
        t = Transcription.from_string("just some text", language="en")
        assert t.text == "just some text"
        assert len(t.chunks) == 1
        assert t.chunks[0].timestamp == (0.0, float(WINDOW_S))

    def test_offset_shifts_timestamps_into_absolute_time(self):
        s = "<|0.00|> one<|1.00|>"
        t = Transcription.from_string(s, language="en", offset=30.0)
        assert t.chunks[0].timestamp == (30.0, 31.0)


class TestAdjustTimestamps:
    def test_zero_offset_is_noop(self):
        t = Transcription(
            language="en",
            text="x",
            chunks=[{"text": "x", "timestamp": (1.0, 2.0)}],
        )
        t.adjust_timestamps(0.0)
        assert t.chunks[0].timestamp == (1.0, 2.0)


class TestMergeOverlapping:
    def test_empty(self):
        t = merge_overlapping([], language="en", overlap_s=5.0)
        assert t.text == ""
        assert t.chunks == []

    def test_single_window_passthrough(self):
        w = Transcription(
            language="en",
            text="hello",
            chunks=[{"text": "hello", "timestamp": (0.0, 2.0)}],
        )
        t = merge_overlapping([w], language="en", overlap_s=5.0)
        assert t.text == "hello"
        assert len(t.chunks) == 1

    def test_dedupes_boundary_segment_keeping_interior_copy(self):
        # Two windows, stride 25s, overlap 5s. Overlap region [25, 30],
        # seam = 27.5. The boundary word "b" is decoded by both windows:
        # as window 0's truncated tail (" b-trunc", at the very edge) and as
        # window 1's clean interior copy (" b"). The merge must keep exactly
        # one copy, and it must be window 1's interior copy. Distinct text on
        # each copy lets the assertion prove *which* one survived, not merely
        # that dedup happened.
        w0 = Transcription(
            language="en",
            text=" a b-trunc",
            chunks=[
                {"text": " a", "timestamp": (10.0, 11.0)},
                {"text": " b-trunc", "timestamp": (29.0, 29.8)},  # > seam, dropped
            ],
        )
        w1 = Transcription(
            language="en",
            text=" b c",
            chunks=[
                {"text": " b", "timestamp": (29.0, 29.8)},  # interior copy, kept
                {"text": " c", "timestamp": (40.0, 41.0)},
            ],
        )
        t = merge_overlapping([w0, w1], language="en", overlap_s=5.0)
        texts = [c.text for c in t.chunks]
        assert texts == [" a", " b", " c"], texts
        assert t.text == " a b c"

    def test_keeps_pre_seam_from_earlier_window(self):
        # A segment before the seam stays with window 0; one after goes to
        # window 1. Distinct text proves the split happened at the seam.
        w0 = Transcription(
            language="en",
            text=" early0",
            chunks=[{"text": " early0", "timestamp": (26.0, 27.0)}],  # < seam 27.5
        )
        w1 = Transcription(
            language="en",
            text=" late1",
            chunks=[{"text": " late1", "timestamp": (28.0, 29.0)}],  # > seam
        )
        t = merge_overlapping([w0, w1], language="en", overlap_s=5.0)
        assert [c.text for c in t.chunks] == [" early0", " late1"]


class TestBatchIterator:
    def test_overlapping_window_count(self):
        # 60s of audio, 30s windows, 5s overlap => stride 25s.
        # Window starts at 0, 25, 50 => 3 windows.
        array = np.zeros(60 * 16000, dtype=np.float32)
        windows = [w for batch in BatchIterator(array, 1, 5.0) for w in batch]
        assert len(windows) == 3

    def test_batches_respect_batch_size(self):
        array = np.zeros(60 * 16000, dtype=np.float32)
        batches = list(BatchIterator(array, batch_size=2, overlap_s=5.0))
        assert [len(b) for b in batches] == [2, 1]

    def test_first_window_is_full_length(self):
        array = np.zeros(60 * 16000, dtype=np.float32)
        first = next(iter(BatchIterator(array, 1, 5.0)))[0]
        assert len(first) == 30 * 16000
