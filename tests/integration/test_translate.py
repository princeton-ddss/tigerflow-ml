"""Integration tests for Translate task."""

import pytest
from tigerflow_ml.text.translate._base import _TranslateBase

_context = None


@pytest.mark.dependency()
def test_setup(make_context):
    global _context
    _context = make_context(_TranslateBase.Params, "translate")
    _TranslateBase.setup(_context)


@pytest.mark.dependency(depends=["test_setup"])
def test_run(translate_dir, get_input_files, make_output_path):
    for input_file in get_input_files(translate_dir):
        output_file = make_output_path(translate_dir, input_file, ".txt")
        _TranslateBase.run(_context, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"
