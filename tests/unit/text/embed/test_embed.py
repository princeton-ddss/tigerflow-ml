"""Unit tests for embed._base."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tigerflow_ml.text.embed._base import _EmbedBase
from tigerflow_ml.utils import EmptyFileError


def _make_context(**kwargs):
    defaults = dict(
        model="test-model",
        revision="main",
        cache_dir=None,
        device="auto",
        allow_fetch=False,
        seed=42,
        per_line=False,
        batch_size=32,
        prompt=None,
        prompt_name=None,
        normalize=False,
        truncate_dim=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestSetup:
    def test_prompt_and_prompt_name_mutually_exclusive(self):
        context = _make_context(prompt="query: ", prompt_name="query")
        with pytest.raises(ValueError, match="mutually exclusive"):
            _EmbedBase.setup(context)

    def test_model_not_found_without_allow_fetch_raises(self):
        context = _make_context(allow_fetch=False)
        with pytest.raises(RuntimeError, match="not found in cache"):
            _EmbedBase.setup(context)


class TestRun:
    def _run(self, tmp_path, content, **context_kwargs):
        context = _make_context(**context_kwargs)
        context.embedder = MagicMock()
        context.embedder.encode.return_value = np.zeros((3, 4))
        context.embedder.encode_document.return_value = np.zeros(4)

        input_file = tmp_path / "input.txt"
        input_file.write_text(content)
        output_file = tmp_path / "output.npy"

        _EmbedBase.run(context, input_file, output_file)
        return context, output_file

    def test_unsupported_output_extension_raises(self, tmp_path):
        context = _make_context()
        context.embedder = MagicMock()
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello")
        output_file = tmp_path / "output.json"

        with pytest.raises(ValueError, match="not a supported output"):
            _EmbedBase.run(context, input_file, output_file)

    def test_empty_file_raises(self, tmp_path):
        context = _make_context()
        context.embedder = MagicMock()
        input_file = tmp_path / "input.txt"
        input_file.write_text("   \n  ")
        output_file = tmp_path / "output.npy"

        with pytest.raises(EmptyFileError):
            _EmbedBase.run(context, input_file, output_file)

    def test_whole_file_uses_encode_document(self, tmp_path):
        context, output_file = self._run(tmp_path, "hello world")

        context.embedder.encode_document.assert_called_once()
        context.embedder.encode.assert_not_called()
        args, _ = context.embedder.encode_document.call_args
        assert args[0] == "hello world"
        assert output_file.exists()

    def test_per_line_uses_encode_with_batch_size(self, tmp_path):
        context, _ = self._run(
            tmp_path,
            "line one\n\nline two\n  \nline three",
            per_line=True,
            batch_size=8,
        )

        context.embedder.encode.assert_called_once()
        context.embedder.encode_document.assert_not_called()
        args, kwargs = context.embedder.encode.call_args
        assert args[0] == ["line one", "line two", "line three"]
        assert kwargs["batch_size"] == 8

    def test_encode_kwargs_omit_unset_optional_fields(self, tmp_path):
        context, _ = self._run(tmp_path, "hello")

        _, kwargs = context.embedder.encode_document.call_args
        assert "prompt" not in kwargs
        assert "prompt_name" not in kwargs
        assert "truncate_dim" not in kwargs
        assert kwargs["normalize_embeddings"] is False

    def test_encode_kwargs_include_set_optional_fields(self, tmp_path):
        context, _ = self._run(
            tmp_path,
            "hello",
            prompt="query: ",
            truncate_dim=128,
            normalize=True,
        )

        _, kwargs = context.embedder.encode_document.call_args
        assert kwargs["prompt"] == "query: "
        assert kwargs["truncate_dim"] == 128
        assert kwargs["normalize_embeddings"] is True
        assert "prompt_name" not in kwargs

    def test_output_written_as_npy(self, tmp_path):
        _, output_file = self._run(tmp_path, "hello world")

        loaded = np.load(output_file)
        assert loaded.shape == (4,)

    def test_per_line_output_written_as_npy(self, tmp_path):
        _, output_file = self._run(tmp_path, "one\ntwo\nthree", per_line=True)

        loaded = np.load(output_file)
        assert loaded.shape == (3, 4)
