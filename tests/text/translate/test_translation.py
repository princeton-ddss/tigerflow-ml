"""Tests for the orchestration module, focused on retry logic."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tigerflow_ml.text.translate._base import (
    _translate_chunk_with_retry,
    _translate_text,
)
from tigerflow_ml.text.translate.translator import (
    ChatTranslator,
    GemmaTranslator,
    build_translator,
)
from tigerflow_ml.text.translate.utils import TranslationError


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that counts whitespace-separated words as tokens."""
    tokenizer = MagicMock()

    def mock_encode(text, add_special_tokens=True):
        return list(range(len(text.split())))

    def mock_decode(token_ids):
        return f"[{len(token_ids)} tokens]"

    tokenizer.encode = mock_encode
    tokenizer.decode = mock_decode
    return tokenizer


@pytest.fixture
def mock_translator(mock_tokenizer):
    translator = MagicMock()
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = 10
    translator.batch_size = 4
    return translator


class TestTranslateChunkWithRetry:
    def test_success_on_first_attempt(self, mock_tokenizer, mock_translator):
        """No truncation: translate() called once, result returned directly."""
        mock_translator.translate.return_value = "translated text"

        result = _translate_chunk_with_retry(
            "some text",
            mock_translator,
            "de",
            "en",
            mock_tokenizer,
            max_tokens=10,
            max_retries=3,
        )

        assert result == "translated text"
        mock_translator.translate.assert_called_once_with("some text", "de", "en")

    def test_truncation_triggers_sub_chunk_retry(self, mock_tokenizer, mock_translator):
        """Truncation on first call → text splits into sub-chunks that succeed."""
        mock_translator.translate.side_effect = [
            TranslationError("output truncated (hit 10 token limit)"),
            "first part",
            "second part",
        ]

        # 6 tokens total; 60% split → max_tokens=3; two 3-token paragraphs split cleanly
        text = "Word one two.\n\nWord three four."
        result = _translate_chunk_with_retry(
            text,
            mock_translator,
            "de",
            "en",
            mock_tokenizer,
            max_tokens=10,
            max_retries=3,
        )

        assert result == "first part\n\nsecond part"
        assert mock_translator.translate.call_count == 3  # 1 failed + 2 sub-chunks

    def test_non_truncation_error_propagates_immediately(
        self, mock_tokenizer, mock_translator
    ):
        """A non-truncation TranslationError must not trigger retry."""
        mock_translator.translate.side_effect = TranslationError("empty result")

        with pytest.raises(TranslationError, match="empty result"):
            _translate_chunk_with_retry(
                "some text",
                mock_translator,
                "de",
                "en",
                mock_tokenizer,
                max_tokens=10,
                max_retries=3,
            )

        mock_translator.translate.assert_called_once()

    def test_exceeds_max_retries_raises(self, mock_tokenizer, mock_translator):
        """Persistent truncation after max_retries raises TranslationError."""
        mock_translator.translate.side_effect = TranslationError("output truncated")

        with pytest.raises(TranslationError, match="retry"):
            _translate_chunk_with_retry(
                "Word one two.\n\nWord three four.",
                mock_translator,
                "de",
                "en",
                mock_tokenizer,
                max_tokens=10,
                max_retries=1,
            )


class TestTranslateText:
    def test_short_text_uses_single_translate(self, mock_tokenizer, mock_translator):
        """Text within token limit bypasses translate_batch entirely."""
        mock_translator.translate.return_value = "translated"

        result = _translate_text("short text", mock_translator, "de", "en")

        assert result == "translated"
        mock_translator.translate.assert_called_once()
        mock_translator.translate_batch.assert_not_called()

    def test_long_text_uses_translate_batch(self, mock_tokenizer, mock_translator):
        """Text exceeding token limit is chunked and sent to translate_batch."""
        mock_translator.max_chunk_tokens = 3
        mock_translator.translate_batch.return_value = [
            "chunk one",
            "chunk two",
            "chunk three",
        ]

        # 9 words → 3 paragraphs of 3 tokens each; each splits at max_tokens=3
        text = "one two three.\n\nfour five six.\n\nseven eight nine."
        result = _translate_text(text, mock_translator, "de", "en")

        assert result == "chunk one\n\nchunk two\n\nchunk three"
        mock_translator.translate_batch.assert_called_once()
        mock_translator.translate.assert_not_called()

    def test_batch_non_truncation_error_propagates(
        self, mock_tokenizer, mock_translator
    ):
        """Non-truncation error from translate_batch must propagate without retrying."""
        mock_translator.max_chunk_tokens = 3
        mock_translator.translate_batch.side_effect = TranslationError("empty result")

        text = "one two three.\n\nfour five six."
        with pytest.raises(TranslationError, match="empty result"):
            _translate_text(text, mock_translator, "de", "en")

        mock_translator.translate.assert_not_called()


def _make_tgemma_hf_translator(mock_tokenizer, max_chunk_tokens=5, batch_size=4):
    """Instantiate HuggingFaceTranslator without loading a model."""
    translator = object.__new__(GemmaTranslator)
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = max_chunk_tokens
    translator.batch_size = batch_size
    translator.pipe = MagicMock()
    return translator


def _make_chat_hf_translator(mock_tokenizer, max_chunk_tokens=5, batch_size=4):
    """Instantiate HuggingFaceTranslator without loading a model."""
    translator = object.__new__(ChatTranslator)
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = max_chunk_tokens
    translator.batch_size = batch_size
    translator.prompt_template = "Translate from {source_lang} to {target_lang}."
    translator.pipe = MagicMock()
    translator._is_vlm = False
    translator._has_chat_template = False
    return translator


def _pipe_batch_output(*texts):
    """HF pipeline return value for a batch call: list of N wrapped outputs."""
    return [
        [
            {
                "generated_text": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": t},
                ]
            }
        ]
        for t in texts
    ]


def _pipe_single_output(text):
    """HF pipeline return value for a single-message call: one-item list of dicts."""
    return [
        {
            "generated_text": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": text},
            ]
        }
    ]


def _pipe_plain_output(text):
    """HF text-generation output (no chat template): plain generated_text string."""
    return [{"generated_text": text}]


class TestTranslateBatchRetryTgemmaBackend:
    def test_only_truncated_chunk_is_retried(self, mock_tokenizer):
        """Truncated chunk triggers retry; other chunks pass through unmodified."""
        translator = _make_tgemma_hf_translator(mock_tokenizer, max_chunk_tokens=5)

        # Batch call returns chunk 2 as truncated (6 words ≥ max_chunk_tokens=5)
        truncated_output = "a b c d e f"  # 6 tokens → is_truncated() == True
        translator.pipe.side_effect = [
            _pipe_batch_output(
                "translated one", truncated_output, "translated three"
            ),  # batch
            _pipe_single_output("retried two a"),  # sub-chunk 1 of chunk 2
            _pipe_single_output("retried two b"),  # sub-chunk 2 of chunk 2
        ]

        # chunk 2 is "Word one two.\n\nWord three four." (6 tokens); 60% split → max=3
        # → splits cleanly into two 3-token paragraphs for the retry
        results = translator.translate_batch(
            ["chunk one", "Word one two.\n\nWord three four.", "chunk three"],
            "de",
            "en",
        )

        assert results[0] == "translated one"
        assert results[1] == "retried two a\n\nretried two b"
        assert results[2] == "translated three"
        assert translator.pipe.call_count == 3  # 1 batch + 2 sub-chunk retries

    def test_no_retry_when_no_truncation(self, mock_tokenizer):
        """All chunks fit → pipe called once, no retry calls."""
        translator = _make_tgemma_hf_translator(mock_tokenizer, max_chunk_tokens=5)
        translator.pipe.return_value = _pipe_batch_output("one", "two")

        results = translator.translate_batch(["chunk one", "chunk two"], "de", "en")

        assert results == ["one", "two"]
        assert translator.pipe.call_count == 1

    def test_retry_raises_after_max_depth(self, mock_tokenizer):
        """Persistent truncation in retry sub-chunks raises after MAX_DEPTH attempts."""
        translator = _make_tgemma_hf_translator(mock_tokenizer, max_chunk_tokens=5)

        # Every pipe call returns a truncated result
        always_truncated = "a b c d e f"  # 6 tokens ≥ 5
        translator.pipe.side_effect = [
            _pipe_batch_output(always_truncated),  # initial batch call
            _pipe_single_output(always_truncated),  # retry depth 1, sub-chunk 1
            _pipe_single_output(always_truncated),  # retry depth 1, sub-chunk 2
            _pipe_single_output(always_truncated),  # retry depth 2, sub-chunk 1
            _pipe_single_output(always_truncated),  # retry depth 2, sub-chunk 2
            _pipe_single_output(always_truncated),  # retry depth 2, sub-chunk 1
            _pipe_single_output(always_truncated),  # retry depth 2, sub-chunk 2
        ]

        with pytest.raises(TranslationError, match="retry"):
            translator.translate_batch(
                ["Word one two.\n\nWord three four."], "de", "en"
            )


class TestTranslateBatchRetryChatBackend:
    """ChatTranslator: sequential translate_batch, one pipe call per chunk."""

    def test_only_truncated_chunk_is_retried(self, mock_tokenizer):
        """Truncated chunk triggers retry; other chunks pass through unmodified."""
        translator = _make_chat_hf_translator(mock_tokenizer, max_chunk_tokens=5)

        truncated_output = "a b c d e f"  # 6 tokens → is_truncated() == True
        # Sequential: one pipe call per chunk; chunk 2 truncated → 2 sub-chunk retries
        translator.pipe.side_effect = [
            _pipe_plain_output("translated one"),  # chunk 1
            _pipe_plain_output(truncated_output),  # chunk 2 → truncated
            _pipe_plain_output("retried two a"),  # chunk 2 sub-chunk 1
            _pipe_plain_output("retried two b"),  # chunk 2 sub-chunk 2
            _pipe_plain_output("translated three"),  # chunk 3
        ]

        # chunk 2 is "Word one two.\n\nWord three four." (6 tokens); 60% split → max=3
        # → splits cleanly into two 3-token paragraphs for the retry
        results = translator.translate_batch(
            ["chunk one", "Word one two.\n\nWord three four.", "chunk three"],
            "de",
            "en",
        )

        assert results[0] == "translated one"
        assert results[1] == "retried two a\n\nretried two b"
        assert results[2] == "translated three"
        assert translator.pipe.call_count == 5  # 3 direct + 2 sub-chunk retries

    def test_no_retry_when_no_truncation(self, mock_tokenizer):
        """All chunks fit → pipe called once per chunk, no retries."""
        translator = _make_chat_hf_translator(mock_tokenizer, max_chunk_tokens=5)
        translator.pipe.side_effect = [
            _pipe_plain_output("one"),
            _pipe_plain_output("two"),
        ]

        results = translator.translate_batch(["chunk one", "chunk two"], "de", "en")

        assert results == ["one", "two"]
        assert translator.pipe.call_count == 2

    def test_retry_raises_after_max_depth(self, mock_tokenizer):
        """Persistent truncation exhausts MAX_DEPTH retry levels and raises."""
        translator = _make_chat_hf_translator(mock_tokenizer, max_chunk_tokens=5)

        # Every pipe call returns a truncated result regardless of input size
        always_truncated = "a b c d e f"  # 6 tokens ≥ 5
        translator.pipe.side_effect = [_pipe_plain_output(always_truncated)] * 10

        with pytest.raises(TranslationError, match="retry"):
            translator.translate_batch(
                ["Word one two.\n\nWord three four."], "de", "en"
            )


class TestBuildTranslatorFactory:
    def test_tgemma_backend_returns_gemma_translator(self, mock_tokenizer):
        with patch.object(GemmaTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=SimpleNamespace(),
                backend="tgemma",
            )
        assert isinstance(translator, GemmaTranslator)

    def test_chat_backend_text_only_model(self, mock_tokenizer):
        with patch.object(ChatTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=SimpleNamespace(),
                backend="chat",
            )
        assert isinstance(translator, ChatTranslator)
        assert translator._is_vlm is False

    def test_chat_backend_vlm_sets_is_vlm(self, mock_tokenizer):
        config = SimpleNamespace(model_type="qwen2_5_vl", vision_config=object())
        with patch.object(ChatTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=config,
                backend="chat",
            )
        assert isinstance(translator, ChatTranslator)
        assert translator._is_vlm is True

    def test_auto_gemma_vlm_returns_gemma_translator(self, mock_tokenizer):
        config = SimpleNamespace(model_type="gemma3", vision_config=object())
        with patch.object(GemmaTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=config,
                backend="auto",
            )
        assert isinstance(translator, GemmaTranslator)

    def test_auto_non_gemma_vlm_returns_chat_translator_with_is_vlm(
        self, mock_tokenizer
    ):
        config = SimpleNamespace(model_type="qwen2_5_vl", vision_config=object())
        with patch.object(ChatTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=config,
                backend="auto",
            )
        assert isinstance(translator, ChatTranslator)
        assert translator._is_vlm is True

    def test_auto_text_only_returns_chat_translator_without_is_vlm(
        self, mock_tokenizer
    ):
        config = SimpleNamespace(model_type="llama")
        with patch.object(ChatTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=config,
                backend="auto",
            )
        assert isinstance(translator, ChatTranslator)
        assert translator._is_vlm is False
