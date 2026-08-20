"""Unit tests for embed._base."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tigerflow_ml.multimodal.embed._base import _EmbedBase
from tigerflow_ml.utils import EmptyFileError, parse_kwargs


def _make_context(**kwargs):
    defaults = dict(
        model="test-model",
        revision="main",
        cache_dir=None,
        device="auto",
        allow_fetch=False,
        seed=42,
        encode_kwargs="{}",
        normalize=False,
        truncate_dim=None,
        use_encode_document=False,
        use_encode_query=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestSetup:
    def test_use_query_and_use_document_mutually_exclusive(self):
        context = _make_context(use_encode_document=True, use_encode_query=True)
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
        context.embedder.encode.return_value = np.zeros(4)
        context.embedder.encode_document.return_value = np.zeros(5)
        context.embedder.encode_query.return_value = np.zeros(6)

        # Mirrors the resolution _EmbedBase.setup() performs on context.encode_kwargs.
        user_encode_kwargs = parse_kwargs(context.encode_kwargs)
        context.encode_kwargs = {
            "normalize_embeddings": context.normalize,
            "show_progress_bar": False,
        }
        if context.truncate_dim is not None:
            context.encode_kwargs["truncate_dim"] = context.truncate_dim
        context.encode_kwargs.update(user_encode_kwargs)

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

    def test_encode_kwargs_omit_unset_optional_fields(self, tmp_path):
        context, _ = self._run(tmp_path, "hello")

        _, kwargs = context.embedder.encode.call_args
        assert "truncate_dim" not in kwargs
        assert kwargs["normalize_embeddings"] is False

    def test_encode_kwargs_include_set_optional_fields(self, tmp_path):
        context, _ = self._run(
            tmp_path,
            "hello",
            truncate_dim=128,
            normalize=True,
        )

        _, kwargs = context.embedder.encode.call_args
        assert kwargs["truncate_dim"] == 128
        assert kwargs["normalize_embeddings"] is True

    def test_output_written_as_npy(self, tmp_path):
        _, output_file = self._run(tmp_path, "hello world")

        loaded = np.load(output_file)
        assert loaded.shape == (4,)

    def test_uses_encode_by_default(self, tmp_path):
        context, output_file = self._run(tmp_path, "hello world")

        context.embedder.encode.assert_called_once()
        context.embedder.encode_document.assert_not_called()
        context.embedder.encode_query.assert_not_called()

        args, _ = context.embedder.encode.call_args
        assert args[0] == "hello world"
        assert output_file.exists()

    def test_uses_encode_document_when_set(self, tmp_path):
        context, output_file = self._run(
            tmp_path, "hello world", use_encode_document=True
        )

        context.embedder.encode_document.assert_called_once()
        context.embedder.encode.assert_not_called()
        context.embedder.encode_query.assert_not_called()

        args, _ = context.embedder.encode_document.call_args
        assert args[0] == "hello world"
        assert output_file.exists()

    def test_uses_encode_query_when_set(self, tmp_path):
        context, output_file = self._run(tmp_path, "hello world", use_encode_query=True)

        context.embedder.encode_query.assert_called_once()
        context.embedder.encode.assert_not_called()
        context.embedder.encode_document.assert_not_called()

        args, _ = context.embedder.encode_query.call_args
        assert args[0] == "hello world"
        assert output_file.exists()
