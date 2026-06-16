"""Integration tests for OCR task."""

import pytest

from tigerflow_ml.text.ocr._base import _OCRBase

pytestmark = pytest.mark.skip(
    reason="OCR task needs refactor before integration test wire-up; tracked in #84"
)

_context = None
_json_context = None


@pytest.mark.dependency()
def test_setup(make_context):
    global _context
    _context = make_context(_OCRBase.Params, "ocr")
    _OCRBase.setup(_context)


@pytest.mark.dependency(depends=["test_setup"])
def test_run(ocr_dir, get_input_files, make_output_path):
    for input_file in get_input_files(ocr_dir):
        output_file = make_output_path(input_file, ".txt")
        _OCRBase.run(_context, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"
