"""Integration tests for Transcribe task.

Audio fixtures live at ``tests/integration/fixtures/transcribe/`` (a small
committed sample), or are supplied via ``TIGERFLOW_ML_TEST_DIR``.
"""

import json

import pytest

from tigerflow_ml.audio.transcribe._base import OutputFormat, _TranscribeBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="module")
def default_context(make_context):
    import gc

    import torch

    ctx = make_context(_TranscribeBase.Params, "transcribe")
    _TranscribeBase.setup(ctx)
    yield ctx
    del ctx.whisper
    del ctx.processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_setup(default_context):
    assert default_context.whisper is not None
    assert default_context.processor is not None


def test_run_text(
    default_context,
    transcribe_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    default_context.output_format = OutputFormat.TEXT
    for input_file in get_input_files(transcribe_dir):
        output_file = make_output_path(input_file, ".txt")
        _TranscribeBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"
        assert_or_update_snapshot(
            text,
            f"transcribe/{input_file.stem}.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )


def test_run_json(default_context, transcribe_dir, get_input_files, make_output_path):
    default_context.output_format = OutputFormat.JSON
    for input_file in get_input_files(transcribe_dir):
        output_file = make_output_path(input_file, ".json")
        _TranscribeBase.run(default_context, input_file, output_file)

        data = json.loads(output_file.read_text(encoding="utf-8"))
        # Schema parity with speech-recognition-inference.
        assert set(data) == {"language", "text", "chunks"}
        assert isinstance(data["text"], str) and data["text"].strip()
        assert isinstance(data["chunks"], list) and data["chunks"]
        for chunk in data["chunks"]:
            assert set(chunk) == {"text", "timestamp"}
            start, end = chunk["timestamp"]
            # Timestamps are always real floats (never null).
            assert isinstance(start, (int, float))
            assert isinstance(end, (int, float))


def test_run_srt(default_context, transcribe_dir, get_input_files, make_output_path):
    default_context.output_format = OutputFormat.SRT
    for input_file in get_input_files(transcribe_dir):
        output_file = make_output_path(input_file, ".srt")
        _TranscribeBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty SRT for {input_file.name}"
        assert "-->" in text, f"No SRT timestamps in {input_file.name}"
        # First cue is numbered 1.
        assert text.lstrip().startswith("1\n")
