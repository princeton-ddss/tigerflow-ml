"""Unit tests for the Whisper transcription engine (no model downloads)."""

from pathlib import Path

import numpy as np
import pytest

from tigerflow_ml.audio.transcribe.transcriber import (
    SAMPLING_RATE,
    WINDOW_S,
    BatchIterator,
    Transcription,
    _flatten_sequences,
    load_audio,
    load_whisper,
    merge_overlapping,
)

_FIXTURE = (
    Path(__file__).parents[3] / "integration" / "fixtures" / "transcribe" / "sample.wav"
)


class TestLoadWhisperErrors:
    def test_missing_model_offline_raises_runtime_error(self):
        # allow_fetch=False on an uncached model hits the OSError -> RuntimeError
        # "not found in cache" path without touching the network.
        with pytest.raises(RuntimeError, match="not found in cache"):
            load_whisper(
                "tigerflow-ml/definitely-not-a-real-model",
                revision="main",
                cache_dir=None,
                allow_fetch=False,
                device="cpu",
                seed=42,
            )


class TestLoadAudioErrors:
    def test_unreadable_file_raises(self, tmp_path):
        bad = tmp_path / "not-audio.wav"
        bad.write_bytes(b"this is not a wav file")
        with pytest.raises(Exception):
            load_audio(bad)


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

    def test_trailing_open_segment_is_recovered(self):
        # Generation truncated mid-segment (e.g. hit max tokens): a final
        # segment opens with a timestamp but has no closing one. It must not be
        # dropped -- it spans from its start to the window end.
        s = "<|0.00|> Hello<|2.50|><|2.50|> cut off at the end"
        t = Transcription.from_string(s, language="en")
        assert t.text == " Hello cut off at the end"
        assert len(t.chunks) == 2
        assert t.chunks[0].timestamp == (0.0, 2.5)
        assert t.chunks[1].text == " cut off at the end"
        assert t.chunks[1].timestamp == (2.5, float(WINDOW_S))

    def test_trailing_open_segment_respects_offset(self):
        s = "<|0.00|> Hello<|2.50|><|2.50|> tail"
        t = Transcription.from_string(s, language="en", offset=30.0)
        assert t.chunks[1].timestamp == (32.5, 30.0 + float(WINDOW_S))

    def test_trailing_open_whitespace_only_is_ignored(self):
        # A bare trailing timestamp with no real text adds no chunk.
        s = "<|0.00|> Hello<|2.50|><|2.50|>   "
        t = Transcription.from_string(s, language="en")
        assert t.text == " Hello"
        assert len(t.chunks) == 1


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
        t = merge_overlapping([], language="en")
        assert t.text == ""
        assert t.chunks == []

    def test_single_window_passthrough(self):
        w = Transcription(
            language="en",
            text="hello",
            chunks=[{"text": "hello", "timestamp": (0.0, 2.0)}],
        )
        t = merge_overlapping([w], language="en")
        assert t.text == "hello"
        assert len(t.chunks) == 1

    def test_skips_chunks_already_covered(self):
        # Window 1's first chunk is fully behind the cursor (the end of
        # window 0's last kept chunk), so it's already transcribed and skipped.
        w0 = Transcription(
            language="en",
            text=" a b",
            chunks=[
                {"text": " a", "timestamp": (10.0, 12.0)},
                {"text": " b", "timestamp": (26.0, 29.0)},  # cursor -> 29.0
            ],
        )
        w1 = Transcription(
            language="en",
            text=" b c",
            chunks=[
                {"text": " b", "timestamp": (26.0, 28.5)},  # end 28.5 <= 29 -> skip
                {"text": " c", "timestamp": (29.0, 31.0)},  # end 31 > 29 -> keep
            ],
        )
        t = merge_overlapping([w0, w1], language="en")
        assert [c.text for c in t.chunks] == [" a", " b", " c"]

    def test_loss_averse_recovers_straddling_span(self):
        # The ireland1 #regression: a clause is decoded by both windows, where
        # window 0 ends mid-clause and only window 1's straddling chunk carries
        # the tail. A seam-cut would orphan it; the loss-averse rule keeps any
        # chunk that extends past the cursor, so the tail survives (with some
        # duplication, which is acceptable for the merged formats).
        w0 = Transcription(
            language="en",
            text=" they act as a",
            chunks=[
                {"text": " they act as a", "timestamp": (97.0, 102.5)},
            ],  # cursor -> 102.5
        )
        w1 = Transcription(
            language="en",
            text=" they act as a prism, division of white light",
            chunks=[
                # straddles the cursor: end 105.3 > 102.5 -> kept (carries tail)
                {
                    "text": " they act as a prism, division of",
                    "timestamp": (100.0, 105.3),
                },
                {"text": " white light", "timestamp": (105.3, 108.0)},
            ],
        )
        t = merge_overlapping([w0, w1], language="en")
        # The "division of" tail must be present (never dropped).
        assert "division of" in t.text
        assert "white light" in t.text


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

    def test_stops_when_window_reaches_end(self):
        # 50s + 1 sample: windows start at 0, 25, 50. The second one [25, 55)
        # already covers the rest of the file, so the third (1-sample) window
        # would decode as padded silence (hallucination). Iteration stops at 2.
        array = np.zeros(50 * 16000 + 1, dtype=np.float32)
        windows = [w for batch in BatchIterator(array, 1, 5.0) for w in batch]
        assert len(windows) == 2

    def test_trailing_window_never_shorter_than_overlap(self):
        # The last window is always longer than the overlap, for any duration --
        # never a degenerate sliver. Checked across awkward boundary lengths.
        for samples in [25 * 16000 + 1, 30 * 16000 + 1, 55 * 16000]:
            array = np.zeros(samples, dtype=np.float32)
            windows = [w for batch in BatchIterator(array, 1, 5.0) for w in batch]
            assert min(len(w) for w in windows) > 5 * 16000
