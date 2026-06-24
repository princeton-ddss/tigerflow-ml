import json
from pathlib import Path

import pytest

from tigerflow_ml.text.ocr._base import (
    OutputFormat,
    _determine_output_format,
    _format_output,
    _validate_output_format,
)


class TestValidateOutputFormat:
    def test_txt_plain_string_passes(self):
        _validate_output_format("hello world", OutputFormat.TEXT)

    def test_txt_empty_string_passes(self):
        _validate_output_format("", OutputFormat.TEXT)

    def test_txt_with_headers_passes(self):
        _validate_output_format("# Title\n\nSome text\n\n## Section", OutputFormat.TEXT)

    def test_valid_json_object_passes(self):
        _validate_output_format('{"key": "value"}', OutputFormat.JSON)

    def test_valid_json_array_passes(self):
        _validate_output_format('[{"page": 1, "text": "hello"}]', OutputFormat.JSON)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _validate_output_format("not json at all", OutputFormat.JSON)

    def test_truncated_json_raises(self):
        with pytest.raises(ValueError):
            _validate_output_format('{"key": "val', OutputFormat.JSON)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _validate_output_format("", OutputFormat.JSON)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            _validate_output_format("some output", "unknown_format")  # type: ignore[arg-type]


class TestFormatOutput:
    def test_json_single_page(self):
        result = _format_output(['{"text": "hello"}'], OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed == [{"text": "hello"}]

    def test_json_multiple_pages(self):
        result = _format_output(
            ['{"text": "page1"}', '{"text": "page2"}'], OutputFormat.JSON
        )
        parsed = json.loads(result)
        assert parsed == [{"text": "page1"}, {"text": "page2"}]

    def test_text_single_page(self):
        assert _format_output(["hello world"], OutputFormat.TEXT) == "hello world"

    def test_text_multiple_pages_joined_with_form_feed(self):
        result = _format_output(["page one", "page two"], OutputFormat.TEXT)
        assert result == "page one\fpage two"

    def test_accepts_generator(self):
        result = _format_output(
            (p for p in ['{"a": 1}', '{"b": 2}']), OutputFormat.JSON
        )
        assert json.loads(result) == [{"a": 1}, {"b": 2}]


class TestDetermineOutputFormat:
    def test_txt(self):
        assert _determine_output_format(Path("out.txt")) == OutputFormat.TEXT
        assert _determine_output_format(Path("out.text")) == OutputFormat.TEXT
        assert _determine_output_format(Path("out.md")) == OutputFormat.TEXT
        assert _determine_output_format(Path("out.markdown")) == OutputFormat.TEXT
        assert _determine_output_format(Path("out.mdown")) == OutputFormat.TEXT
        assert _determine_output_format(Path("out.mkd")) == OutputFormat.TEXT

    def test_json(self):
        assert _determine_output_format(Path("out.json")) == OutputFormat.JSON

    def test_case_insensitive(self):
        assert _determine_output_format(Path("out.TXT")) == OutputFormat.TEXT
        assert _determine_output_format(Path("out.JSON")) == OutputFormat.JSON

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            _determine_output_format(Path("out.pdf"))

    def test_no_extension_raises(self):
        with pytest.raises(ValueError):
            _determine_output_format(Path("out"))
