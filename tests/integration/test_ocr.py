"""Integration tests for OCR task."""

import json

import pytest

from tigerflow_ml.text.ocr._base import OutputFormat, _OCRBase

_context = None
_json_context = None


@pytest.mark.dependency()
def test_setup(make_context):
    global _context
    _context = make_context(_OCRBase.Params, "ocr")
    _OCRBase.setup(_context)


@pytest.mark.dependency()
def test_setup_json_format(make_context):
    global _json_context
    _json_context = make_context(
        _OCRBase.Params, "ocr", output_format=OutputFormat.JSON
    )
    _OCRBase.setup(_json_context)


@pytest.mark.dependency(depends=["test_setup"])
def test_run(ocr_dir, get_input_files, make_output_path):
    for input_file in get_input_files(ocr_dir):
        output_file = make_output_path(ocr_dir, input_file, ".txt")
        _OCRBase.run(_context, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"


@pytest.mark.dependency(depends=["test_setup_json_format"])
def test_run_json_format(ocr_dir, get_input_files, make_output_path):
    for input_file in get_input_files(ocr_dir):
        output_file = make_output_path(ocr_dir, input_file, ".json")
        _OCRBase.run(_json_context, input_file, output_file)

        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert "pages" in data
        assert isinstance(data["pages"], list)
        assert len(data["pages"]) > 0
        for page in data["pages"]:
            assert "page" in page
            assert "text" in page
