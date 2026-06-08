"""Tests for the shared utils module."""

from unittest.mock import MagicMock, patch

import pytest

from tigerflow_ml.utils import (
    ENCODING_FALLBACK_CHAIN,
    ModelConfigParsingError,
    load_model_config,
    read_file_with_fallback,
)


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


class TestLoadModelConfig:
    def test_returns_config_on_success(self):
        mock_config = MagicMock()
        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            result = load_model_config("some/model")
        assert result is mock_config

    def test_passes_args_to_from_pretrained(self):
        mock_config = MagicMock()
        with patch(
            "transformers.AutoConfig.from_pretrained", return_value=mock_config
        ) as mock_fn:
            load_model_config(
                "some/model", allow_fetch=True, cache_dir="/cache", revision="v1"
            )
        mock_fn.assert_called_once_with(
            "some/model",
            local_files_only=False,
            cache_dir="/cache",
            revision="v1",
        )

    def test_oserror_no_fetch_raises_file_not_found(self):
        with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError):
            with pytest.raises(FileNotFoundError, match="some/model"):
                load_model_config("some/model", allow_fetch=False)

    def test_oserror_allow_fetch_raises_config_parsing_error(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=OSError("network error"),
        ):
            with pytest.raises(ModelConfigParsingError):
                load_model_config("some/model", allow_fetch=True)

    def test_unexpected_exception_raises_config_parsing_error(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=ValueError("bad config"),
        ):
            with pytest.raises(ModelConfigParsingError, match="bad config"):
                load_model_config("some/model")
