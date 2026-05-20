"""Tests for the utils module."""

import pytest

from tigerflow_ml.text.translate.utils import (
    ENCODING_FALLBACK_CHAIN,
    SkippedFileError,
    TranslationError,
    read_file_with_fallback,
)


class TestExceptions:
    def test_translation_error_is_exception(self):
        with pytest.raises(TranslationError):
            raise TranslationError("test error")

    def test_skipped_file_error_is_exception(self):
        with pytest.raises(SkippedFileError):
            raise SkippedFileError("test skip")

    def test_exception_messages(self):
        try:
            raise TranslationError("custom message")
        except TranslationError as e:
            assert str(e) == "custom message"


class TestEncodingFallbackChain:
    def test_utf8_sig_is_first(self):
        assert ENCODING_FALLBACK_CHAIN[0] == "utf-8-sig"

    def test_latin1_is_last(self):
        # latin-1 must be last because it accepts all byte values
        assert ENCODING_FALLBACK_CHAIN[-1] == "latin-1"

    def test_no_duplicate_encodings(self):
        # iso-8859-1 and latin-1 are the same in Python, should not have both
        assert "iso-8859-1" not in ENCODING_FALLBACK_CHAIN


class TestReadFileWithFallback:
    def test_reads_utf8_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        content = "Hello, world! Привет мир! 你好世界!"
        file_path.write_text(content, encoding="utf-8")

        result = read_file_with_fallback(file_path)
        assert result == content

    def test_reads_utf8_bom_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        content = "Hello with BOM"
        file_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        result = read_file_with_fallback(file_path)
        assert result == content

    def test_reads_cp1252_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        # CP1252-specific characters (smart quotes, em-dash)
        content = "Hello \u201cworld\u201d \u2014 test"
        file_path.write_bytes(content.encode("cp1252"))

        result = read_file_with_fallback(file_path)
        assert "Hello" in result
        assert "test" in result

    def test_reads_latin1_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        content = "Café résumé naïve"
        file_path.write_bytes(content.encode("latin-1"))

        result = read_file_with_fallback(file_path)
        # Should successfully read (though may decode differently)
        assert len(result) > 0

    def test_nonexistent_file_raises(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            read_file_with_fallback(file_path)

    def test_empty_file_returns_empty_string(self, tmp_path):
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")

        result = read_file_with_fallback(file_path)
        assert result == ""

    def test_file_with_only_whitespace(self, tmp_path):
        file_path = tmp_path / "whitespace.txt"
        file_path.write_text("   \n\n   \t\t   ")

        result = read_file_with_fallback(file_path)
        assert result == "   \n\n   \t\t   "
