"""Tests for the orchestration module, focused on retry logic."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tigerflow_ml.text.translate._base import (
    _DEFAULT_PROMPT,
    _translate_chunk_with_retry,
    _translate_file,
    _translate_text,
    _TranslateBase,
)
from tigerflow_ml.text.translate.translator import (
    GemmaTranslator,
    build_translator,
    get_model_type,
    vllmTranslator,
)
from tigerflow_ml.text.translate.utils import (
    AlreadyInTargetLanguageError,
    EmptyFileError,
    TranslationError,
)


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


class TestTranslateFile:
    def test_empty_file_raises(self, tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        with pytest.raises(EmptyFileError):
            _translate_file(MagicMock(), empty_file, tmp_path / "out.txt")

    def test_already_in_target_lang_raises(self, tmp_path):
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Hello world")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value="en"
        ):
            with pytest.raises(AlreadyInTargetLanguageError):
                _translate_file(
                    MagicMock(), input_file, tmp_path / "out.txt", target_lang="en"
                )

    def test_language_detection_fails_raises(self, tmp_path):
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Hello world")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            with pytest.raises(TranslationError):
                _translate_file(MagicMock(), input_file, tmp_path / "out.txt")


class TestRun:
    @pytest.mark.parametrize(
        "error",
        [
            EmptyFileError("empty"),
            AlreadyInTargetLanguageError("same lang"),
            TranslationError("failed"),
        ],
    )
    def test_run_propagates_translate_file_errors(self, tmp_path, error):
        context = SimpleNamespace(
            translator=MagicMock(), source_lang=None, target_lang="en"
        )
        with patch(
            "tigerflow_ml.text.translate._base._translate_file", side_effect=error
        ):
            with pytest.raises(type(error)):
                _TranslateBase.run(context, tmp_path / "in.txt", tmp_path / "out.txt")


def _make_tgemma_hf_translator(mock_tokenizer, max_chunk_tokens=5, batch_size=4):
    """Instantiate HuggingFaceTranslator without loading a model."""
    translator = object.__new__(GemmaTranslator)
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = max_chunk_tokens
    translator.batch_size = batch_size
    translator.pipe = MagicMock()
    return translator


def _make_vllm_translator(
    mock_tokenizer, max_chunk_tokens=5, system_message="You are an expert linguist"
):
    """Instantiate vllmTranslator without loading a model."""
    translator = object.__new__(vllmTranslator)
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = max_chunk_tokens
    translator.sampling_params = MagicMock()
    translator.prompt_template = _DEFAULT_PROMPT
    translator.system_message = system_message
    translator.model = MagicMock()
    return translator


def _vllm_output(text):
    """Mock vLLM RequestOutput for a single generation."""
    out = MagicMock()
    out.outputs[0].text = text
    return out


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


class TestVllmTranslator:
    """vllmTranslator: uses vLLM's LLM.chat() API directly."""

    def test_translate_returns_generated_text(self, mock_tokenizer):
        """translate() calls model.chat() once and returns the output text."""
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("translated text")]

        result = translator.translate("hello", "de", "en")

        assert result == "translated text"
        translator.model.chat.assert_called_once()

    def test_translate_batch_calls_chat_once(self, mock_tokenizer):
        """translate_batch() sends all messages in one model.chat() call."""
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.return_value = [
            _vllm_output("translated one"),
            _vllm_output("translated two"),
        ]

        results = translator.translate_batch(["text one", "text two"], "de", "en")

        assert results == ["translated one", "translated two"]
        translator.model.chat.assert_called_once()

    def test_translate_passes_sampling_params(self, mock_tokenizer):
        """translate() forwards sampling_params to model.chat()."""
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("ok")]

        translator.translate("text", "de", "en")

        _, kwargs = translator.model.chat.call_args
        assert kwargs.get("sampling_params") is translator.sampling_params

    def test_translate_builds_prompt_from_template(self, mock_tokenizer):
        """translate() interpolates source_lang, target_lang,
        and text into the prompt."""
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("ok")]

        translator.translate("hello world", "de", "en")

        messages = translator.model.chat.call_args[0][0]
        user_content = messages[-1]["content"]
        assert "de" in user_content
        assert "en" in user_content
        assert "hello world" in user_content

    def test_translate_uses_system_message(self, mock_tokenizer):
        """translate() sets the system role content to system_message."""
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("ok")]

        translator.translate("text", "de", "en")

        messages = translator.model.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are an expert linguist"

    def test_translate_uses_custom_system_message(self, mock_tokenizer):
        """translate() reflects a custom system_message in the chat payload."""
        custom_msg = "You are a technical translator specializing in legal documents."
        translator = _make_vllm_translator(mock_tokenizer, system_message=custom_msg)
        translator.model.chat.return_value = [_vllm_output("ok")]

        translator.translate("text", "de", "en")

        messages = translator.model.chat.call_args[0][0]
        assert messages[0]["content"] == custom_msg


class TestGetModelType:
    def test_tgemma_variants(self):
        assert get_model_type("google/translategemma-27b-it") == "tgemma"
        assert get_model_type("google/translategemma-12b-it") == "tgemma"
        assert get_model_type("google/translategemma-2b-it") == "tgemma"

    def test_non_tgemma_returns_chat(self):
        assert get_model_type("Qwen/Qwen2.5-VL-7B-Instruct") == "chat"
        assert get_model_type("meta-llama/Llama-3-8b") == "chat"
        assert get_model_type("google/gemma-3-27b-it") == "chat"


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
                vram_fraction=0.9,
            )
        assert isinstance(translator, GemmaTranslator)

    def test_chat_backend_returns_vllm_translator(self, mock_tokenizer):
        with patch.object(vllmTranslator, "__init__", return_value=None):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=SimpleNamespace(),
                backend="chat",
                vram_fraction=0.9,
            )
        assert isinstance(translator, vllmTranslator)

    def test_auto_tgemma_name_returns_gemma_translator(self, mock_tokenizer):
        with patch.object(GemmaTranslator, "_load_pipeline", return_value=MagicMock()):
            translator = build_translator(
                "google/translategemma-27b-it",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=SimpleNamespace(),
                backend="auto",
                vram_fraction=0.9,
            )
        assert isinstance(translator, GemmaTranslator)

    def test_auto_non_tgemma_returns_vllm_translator(self, mock_tokenizer):
        with patch.object(vllmTranslator, "__init__", return_value=None):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                batch_size=1,
                fetch=False,
                config=SimpleNamespace(),
                backend="auto",
                vram_fraction=0.9,
            )
        assert isinstance(translator, vllmTranslator)
