"""Tests for the orchestration module, focused on retry logic."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tigerflow_ml.text.translate._base import (
    _DEFAULT_PROMPT,
    _FALLBACK_PROMPT,
    _translate_chunk_with_retry,
    _translate_file,
    _translate_text,
    _TranslateBase,
)
from tigerflow_ml.text.translate.translator import (
    TgemmaTranslator,
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
    translator.prompt_template = _DEFAULT_PROMPT
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

    def test_language_detection_fails_raises(self, mock_translator, tmp_path):
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Hello world")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            with pytest.raises(TranslationError):
                _translate_file(mock_translator, input_file, tmp_path / "out.txt")


class TestTranslateFileAutoLangDetect:
    """_translate_file: auto_lang_detect / source_lang interaction."""

    @pytest.fixture
    def translator(self, mock_translator):
        mock_translator.translate.return_value = "translated content"
        return mock_translator

    def test_source_lang_overrides_detected(self, translator, tmp_path):
        """Explicit --source-lang wins over auto-detected language."""
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Bonjour le monde")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value="es"
        ):
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                source_lang="fr",
                target_lang="en",
            )
        assert translator.translate.call_args.args[1] == "fr"

    def test_detection_failure_with_source_lang_succeeds(self, translator, tmp_path):
        """Detection failure is non-fatal when --source-lang is explicitly set."""
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Bonjour le monde")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                source_lang="fr",
                target_lang="en",
            )
        translator.translate.assert_called_once()
        assert translator.translate.call_args.args[1] == "fr"

    def test_auto_lang_detect_disabled_skips_detection(self, translator, tmp_path):
        """auto_lang_detect=False never calls detect_language."""
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Bonjour le monde")
        with patch("tigerflow_ml.text.translate._base.detect_language") as mock_detect:
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                source_lang="fr",
                target_lang="en",
                auto_lang_detect=False,
            )
        mock_detect.assert_not_called()

    def test_auto_lang_detect_disabled_no_source_lang_raises(
        self, translator, tmp_path
    ):
        """auto_lang_detect=False + no source_lang -> TranslationError."""
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Bonjour le monde")
        with pytest.raises(TranslationError, match="--auto-lang-detect"):
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                auto_lang_detect=False,
            )

    def test_source_lang_equals_target_raises_even_when_detected_differs(
        self, translator, tmp_path
    ):
        """Explicit source_lang wins, so source_lang==target_lang trips the check."""
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Hello world")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value="fr"
        ):
            with pytest.raises(AlreadyInTargetLanguageError):
                _translate_file(
                    translator,
                    input_file,
                    tmp_path / "out.txt",
                    source_lang="en",
                    target_lang="en",
                )

    def test_explicit_source_lang_avoids_already_in_target_when_detected_matches(
        self, translator, tmp_path
    ):
        """Detected=='en' would normally raise, but user said --source-lang=fr."""
        input_file = tmp_path / "doc.txt"
        input_file.write_text("ambiguous")
        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value="en"
        ):
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                source_lang="fr",
                target_lang="en",
            )
        assert translator.translate.call_args.args[1] == "fr"


class TestTranslateFileFallback:
    """_translate_file: --use-fallback-prompt routing."""

    def test_fallback_prompt_reaches_model_when_detection_fails(
        self, mock_tokenizer, tmp_path
    ):
        """When detection fails and fallback is enabled, the model.chat
        payload uses _FALLBACK_PROMPT (not the default {source_lang} prompt).
        """
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("translated")]
        input_file = tmp_path / "in.txt"
        input_file.write_text("xyz")

        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                target_lang="en",
                use_fallback_prompt=True,
            )

        sent = translator.model.chat.call_args[0][0][-1]["content"]
        assert sent == _FALLBACK_PROMPT.format(target_lang="en", text="xyz")

    def test_prompt_template_restored_after_translation_exception(
        self, mock_tokenizer, tmp_path
    ):
        """try/finally restores prompt_template if translation raises.

        Note: this test exercises the mutate-restore pattern in
        _translate_file and should be removed alongside that pattern in the
        follow-up refactor that threads prompt_template through translate().
        """
        translator = _make_vllm_translator(mock_tokenizer)
        translator.model.chat.side_effect = RuntimeError("boom")
        input_file = tmp_path / "in.txt"
        input_file.write_text("xyz")

        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            with pytest.raises(RuntimeError):
                _translate_file(
                    translator,
                    input_file,
                    tmp_path / "out.txt",
                    target_lang="en",
                    use_fallback_prompt=True,
                )

        assert translator.prompt_template == _DEFAULT_PROMPT

    def test_no_source_lang_in_template_translates_without_fallback(
        self, mock_tokenizer, tmp_path
    ):
        """When the active template doesn't reference {source_lang},
        detection failure is non-fatal and the custom template (not the
        fallback) reaches the model."""
        custom_template = "Render in {target_lang}: {text}"
        translator = _make_vllm_translator(mock_tokenizer)
        translator.prompt_template = custom_template
        translator.model.chat.return_value = [_vllm_output("translated")]
        input_file = tmp_path / "in.txt"
        input_file.write_text("xyz")

        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            _translate_file(
                translator,
                input_file,
                tmp_path / "out.txt",
                target_lang="en",
                use_fallback_prompt=False,
            )

        sent = translator.model.chat.call_args[0][0][-1]["content"]
        assert sent == custom_template.format(target_lang="en", text="xyz")
        assert translator.prompt_template == custom_template


class TestTranslateFileTgemma:
    """_translate_file: tgemma models reject the fallback path."""

    def _make_tgemma(self, mock_tokenizer):
        translator = object.__new__(TgemmaTranslator)
        translator.tokenizer = mock_tokenizer
        translator.max_chunk_tokens = 5
        translator.sampling_params = MagicMock()
        translator.extra_chat_kwargs = {"use_tqdm": False}
        translator.prompt_template = (
            "<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}"
        )
        translator.system_message = None
        translator.model = MagicMock()
        return translator

    def test_tgemma_raises_even_when_use_fallback_prompt_enabled(
        self, mock_tokenizer, tmp_path
    ):
        """tgemma ignores use_fallback_prompt and requires a source language.
        Its prompt_template stays untouched."""
        translator = self._make_tgemma(mock_tokenizer)
        original_prompt = translator.prompt_template
        input_file = tmp_path / "in.txt"
        input_file.write_text("xyz")

        with patch(
            "tigerflow_ml.text.translate._base.detect_language", return_value=None
        ):
            with pytest.raises(TranslationError):
                _translate_file(
                    translator,
                    input_file,
                    tmp_path / "out.txt",
                    target_lang="en",
                    use_fallback_prompt=True,
                )

        assert translator.prompt_template == original_prompt


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
            translator=MagicMock(),
            source_lang=None,
            target_lang="en",
            auto_lang_detect=True,
            use_fallback_prompt=False,
        )
        with patch(
            "tigerflow_ml.text.translate._base._translate_file", side_effect=error
        ):
            with pytest.raises(type(error)):
                _TranslateBase.run(context, tmp_path / "in.txt", tmp_path / "out.txt")


def _make_vllm_translator(
    mock_tokenizer, max_chunk_tokens=5, system_message="You are an expert linguist"
):
    """Instantiate vllmTranslator without loading a model."""
    translator = object.__new__(vllmTranslator)
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = max_chunk_tokens
    translator.sampling_params = MagicMock()
    translator.extra_chat_kwargs = {"use_tqdm": False}
    translator.prompt_template = _DEFAULT_PROMPT
    translator.system_message = system_message
    translator.model = MagicMock()
    return translator


def _vllm_output(text):
    """Mock vLLM RequestOutput for a single generation."""
    out = MagicMock()
    out.outputs[0].text = text
    return out


class TestVllmTranslator:
    def _init(self, mock_tokenizer, config, max_model_len=None, max_chunk_tokens=100):
        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 0  # triggers "or 1" fallback

        mock_llm_cls = MagicMock()
        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls
        mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())

        modules = {
            "torch": mock_torch,
            "vllm": mock_vllm,
            "huggingface_hub": MagicMock(),
        }
        with (
            patch.dict("sys.modules", modules),
            patch("tigerflow_ml.text.translate.translator.logger") as mock_logger,
        ):
            translator = object.__new__(vllmTranslator)
            translator.__init__(
                model_name="any/model",
                config=config,
                seed=42,
                temperature=0.0,
                tokenizer=mock_tokenizer,
                max_model_len=max_model_len,
                max_chunk_tokens=max_chunk_tokens,
                fetch=False,
            )
        return mock_llm_cls, mock_logger

    def test_user_max_model_len_clamped_and_warns_when_exceeds_cap(
        self, mock_tokenizer
    ):
        config = SimpleNamespace(max_position_embeddings=4096)
        llm_cls, mock_logger = self._init(mock_tokenizer, config, max_model_len=10000)
        assert llm_cls.call_args.kwargs["max_model_len"] == 4096
        mock_logger.warning.assert_called_once()
        assert "Clamping" in mock_logger.warning.call_args[0][0]

    def test_user_max_model_len_preserved_when_within_cap(self, mock_tokenizer):
        config = SimpleNamespace(max_position_embeddings=4096)
        llm_cls, _ = self._init(mock_tokenizer, config, max_model_len=2048)
        assert llm_cls.call_args.kwargs["max_model_len"] == 2048

    def test_auto_max_model_len_clamped_to_cap(self, mock_tokenizer):
        # max_chunk_tokens=1000 → int(1000 * 2.5 + 512) = 3012
        config = SimpleNamespace(max_position_embeddings=2048)
        llm_cls, _ = self._init(mock_tokenizer, config, max_chunk_tokens=1000)
        assert llm_cls.call_args.kwargs["max_model_len"] == 2048

    def test_no_cap_attr_uses_computed_model_len(self, mock_tokenizer):
        config = SimpleNamespace()  # no model len cap
        llm_cls, _ = self._init(mock_tokenizer, config, max_chunk_tokens=10000)
        expected = int(10000 * 2.5 + 512)
        assert llm_cls.call_args.kwargs["max_model_len"] == expected

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

    def test_build_message_raises_when_source_lang_none_and_template_requires_it(
        self, mock_tokenizer
    ):
        """Guard the new None input space opened by widening source_lang."""
        translator = _make_vllm_translator(mock_tokenizer)
        assert "{source_lang}" in translator.prompt_template

        with pytest.raises(ValueError, match="source_lang"):
            translator._build_message("hello", None, "en")


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
        with patch.object(vllmTranslator, "__init__", return_value=None):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                fetch=False,
                config=SimpleNamespace(),
                backend="tgemma",
            )
        assert isinstance(translator, TgemmaTranslator)

    def test_chat_backend_returns_vllm_translator(self, mock_tokenizer):
        with patch.object(vllmTranslator, "__init__", return_value=None):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                fetch=False,
                config=SimpleNamespace(),
                backend="chat",
            )
        assert isinstance(translator, vllmTranslator)

    def test_auto_tgemma_name_returns_gemma_translator(self, mock_tokenizer):
        with patch.object(vllmTranslator, "__init__", return_value=None):
            translator = build_translator(
                "google/translategemma-27b-it",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                fetch=False,
                config=SimpleNamespace(),
                backend="auto",
            )
        assert isinstance(translator, TgemmaTranslator)

    def test_auto_non_tgemma_returns_vllm_translator(self, mock_tokenizer):
        with patch.object(vllmTranslator, "__init__", return_value=None):
            translator = build_translator(
                "any/model",
                tokenizer=mock_tokenizer,
                max_chunk_tokens=100,
                fetch=False,
                config=SimpleNamespace(),
                backend="auto",
            )
        assert isinstance(translator, vllmTranslator)


class TestTgemmaTranslator:
    """TgemmaTranslator: uses tgemma special-token prompt format."""

    def _make(self, mock_tokenizer):
        translator = object.__new__(TgemmaTranslator)
        translator.tokenizer = mock_tokenizer
        translator.max_chunk_tokens = 5
        translator.sampling_params = MagicMock()
        translator.extra_chat_kwargs = {"use_tqdm": False}
        translator.prompt_template = (
            "<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}"
        )
        translator.system_message = None
        translator.model = MagicMock()
        return translator

    def test_prompt_uses_tgemma_format(self, mock_tokenizer):
        """translate() formats the prompt with tgemma special tokens."""
        translator = self._make(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("translated")]

        translator.translate("hello", "de", "en")

        messages = translator.model.chat.call_args[0][0]
        content = messages[-1]["content"]
        assert content == "<<<source>>>de<<<target>>>en<<<text>>>hello"

    def test_no_system_message_by_default(self, mock_tokenizer):
        """TgemmaTranslator sends only a user message (no system role)."""
        translator = self._make(mock_tokenizer)
        translator.model.chat.return_value = [_vllm_output("ok")]

        translator.translate("hello", "de", "en")

        messages = translator.model.chat.call_args[0][0]
        assert all(m["role"] != "system" for m in messages)


class TestSetupValidation:
    def test_setup_raises_if_prompt_template_lacks_text_placeholder(self):
        context = SimpleNamespace(
            prompt_template="Translate to {target_lang}.",
            source_lang="fr",
            target_lang="en",
        )
        with pytest.raises(ValueError, match=r"\{text\}"):
            _TranslateBase.setup(context)

    def test_setup_raises_if_source_lang_equals_target_lang(self):
        context = SimpleNamespace(
            prompt_template=_DEFAULT_PROMPT,
            source_lang="en",
            target_lang="en",
        )
        with pytest.raises(ValueError, match="No translation required"):
            _TranslateBase.setup(context)
