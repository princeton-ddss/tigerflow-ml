"""Integration tests for Transcribe task."""

import json

import pytest
from tigerflow_ml.audio.transcribe._base import OutputFormat, _TranscribeBase

_context = None
_json_context = None
_srt_context = None


@pytest.mark.dependency()
def test_setup(make_context):
    global _context
    _context = make_context(_TranscribeBase.Params, "transcribe")
    _TranscribeBase.setup(_context)


@pytest.mark.dependency()
def test_setup_json_format(make_context):
    global _json_context
    _json_context = make_context(_TranscribeBase.Params, "transcribe", output_format=OutputFormat.JSON)
    _TranscribeBase.setup(_json_context)


@pytest.mark.dependency()
def test_setup_srt_format(make_context):
    global _srt_context
    _srt_context = make_context(_TranscribeBase.Params, "transcribe", output_format=OutputFormat.SRT)
    _TranscribeBase.setup(_srt_context)


@pytest.mark.dependency(depends=["test_setup"])
def test_run(transcribe_dir, get_input_files, make_output_path):
    for input_file in get_input_files(transcribe_dir):
        output_file = make_output_path(transcribe_dir, input_file, ".txt")
        _TranscribeBase.run(_context, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"


@pytest.mark.dependency(depends=["test_setup_json_format"])
def test_run_json_format(transcribe_dir, get_input_files, make_output_path):
    for input_file in get_input_files(transcribe_dir):
        output_file = make_output_path(transcribe_dir, input_file, ".json")
        _TranscribeBase.run(_json_context, input_file, output_file)

        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"].strip()) > 0


@pytest.mark.dependency(depends=["test_setup_srt_format"])
def test_run_srt_format(transcribe_dir, get_input_files, make_output_path):
    for input_file in get_input_files(transcribe_dir):
        output_file = make_output_path(transcribe_dir, input_file, ".srt")
        _TranscribeBase.run(_srt_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty SRT for {input_file.name}"
        assert "-->" in text, f"No SRT timestamps in {input_file.name}"
