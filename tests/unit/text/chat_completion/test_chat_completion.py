"""Unit tests for chat_completion._base."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

import PIL.Image
import pytest

from tigerflow_ml.text.chat_completion._base import (
    _build_img_message,
    _build_txt_message,
    _ChatCompletionBase,
)
from tigerflow_ml.utils import EmptyFileError


def _make_context(**kwargs):
    defaults = dict(
        model="test-model",
        prompt="Describe this",
        system_message=None,
        max_image_pixels=None,
        max_tokens=512,
        max_model_len=32000,
        allow_fetch=False,
        cache_dir="",
        revision="main",
        device="auto",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_pil_image(width, height, color="red"):
    return PIL.Image.new("RGB", (width, height), color=color)


def _save_image(path, width, height, color="red"):
    _make_pil_image(width, height, color).save(path)
    return path


class TestImageResize:
    def _run(self, tmp_path, width, height, max_image_pixels=1024):
        """Run _process_img_file and return the image size passed
        to _build_img_message."""
        img_path = _save_image(tmp_path / "img.png", width, height)
        context = _make_context(max_image_pixels=max_image_pixels)
        captured = {}

        def capture(prompt, image, system_message):
            captured["size"] = image.size
            return _build_img_message(prompt, image, system_message)

        with patch(
            "tigerflow_ml.text.chat_completion._base._build_img_message",
            side_effect=capture,
        ):
            with patch(
                "tigerflow_ml.text.chat_completion._base._run_chat", return_value=""
            ):
                _ChatCompletionBase._process_img_file(context, img_path)

        return captured["size"]

    def test_image_is_not_resized_by_default(self, tmp_path):
        w, h = self._run(tmp_path, 5000, 6000, max_image_pixels=None)
        w == 5000
        h == 6000

    def test_large_image_is_resized(self, tmp_path):
        w, h = self._run(tmp_path, 2000, 1500)
        assert max(w, h) <= 1024

    def test_small_image_is_not_resized(self, tmp_path):
        assert self._run(tmp_path, 100, 80) == (100, 80)

    def test_image_at_exact_limit_is_not_resized(self, tmp_path):
        assert self._run(tmp_path, 1024, 768) == (1024, 768)

    def test_aspect_ratio_preserved(self, tmp_path):
        w, h = self._run(tmp_path, 2000, 500)
        assert w == 1024
        assert w / h == 2000 / 500


class TestSetup:
    def test_max_tokens_exceeds_max_model_len_raises(self):
        context = _make_context(max_tokens=512, max_model_len=256)
        with pytest.raises(ValueError, match="max_tokens"):
            _ChatCompletionBase.setup(context)

    def test_max_tokens_equal_to_max_model_len_raises(self):
        context = _make_context(max_tokens=512, max_model_len=512)
        with pytest.raises(ValueError, match="max_tokens"):
            _ChatCompletionBase.setup(context)

    def test_model_not_found_without_allow_fetch_exits(self):
        context = _make_context(allow_fetch=False)
        with pytest.raises(RuntimeError, match="1"):
            _ChatCompletionBase.setup(context)


class TestRun:
    def test_unsupported_extension_raises_skipped(self, tmp_path):
        context = _make_context()
        input_file = tmp_path / "file.xyz"
        input_file.write_text("content")

        with pytest.raises(ValueError, match="not currently supported"):
            _ChatCompletionBase.run(context, input_file, tmp_path / "out.txt")

    def test_empty_text_file_raises_skipped(self, tmp_path):
        context = _make_context()
        input_file = tmp_path / "file.txt"
        input_file.write_text("   \n  ")

        with pytest.raises(EmptyFileError, match="Empty file"):
            _ChatCompletionBase.run(context, input_file, tmp_path / "out.txt")

    def test_whitespace_only_text_file_raises_skipped(self, tmp_path):
        context = _make_context()
        input_file = tmp_path / "file.txt"
        input_file.write_text("\t\n\r\n")

        with pytest.raises(EmptyFileError, match="Empty file"):
            _ChatCompletionBase.run(context, input_file, tmp_path / "out.txt")


class TestBuildTxtMessage:
    def test_text_placeholder_is_substituted(self):
        messages = _build_txt_message(
            prompt="Summarize: {text}", text="the content", system_message=None
        )
        assert messages[-1]["content"] == "Summarize: the content"

    def test_no_placeholder_appends_content_after_newline(self):
        messages = _build_txt_message(
            prompt="Summarize this:", text="the content", system_message=None
        )
        assert messages[-1]["content"] == "Summarize this:\nthe content"

    def test_system_message_is_first(self):
        messages = _build_txt_message(
            prompt="Hello {text}", text="world", system_message="You are helpful."
        )
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1]["role"] == "user"

    def test_no_system_message_single_user_turn(self):
        messages = _build_txt_message(
            prompt="Hello {text}", text="world", system_message=None
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_unknown_placeholder_raises_value_error(self):
        # {text} must be present to enter the format() path; {name} is then unknown
        with pytest.raises(ValueError, match="unknown placeholder"):
            _build_txt_message(
                prompt="Hello {text} and {name}", text="world", system_message=None
            )


class TestBuildImgMessage:
    @pytest.fixture
    def image(self):
        return _make_pil_image(64, 48, color="blue")

    def _user_content(self, messages):
        return messages[-1]["content"]

    def test_image_comes_before_text(self, image):
        content = self._user_content(
            _build_img_message("Describe", image, system_message=None)
        )
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"

    def test_system_message_is_first(self, image):
        messages = _build_img_message("Describe", image, system_message="Be concise.")
        assert messages[0] == {"role": "system", "content": "Be concise."}
        assert messages[1]["role"] == "user"

    def test_no_system_message_single_user_turn(self, image):
        messages = _build_img_message("Describe", image, system_message=None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_base64_url_is_valid_png(self, image):
        content = self._user_content(
            _build_img_message("Describe", image, system_message=None)
        )
        url = next(c for c in content if c["type"] == "image_url")["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        raw = base64.b64decode(url.removeprefix("data:image/png;base64,"))
        decoded = PIL.Image.open(io.BytesIO(raw))
        assert decoded.format == "PNG"

    def test_base64_image_has_correct_dimensions(self, image):
        content = self._user_content(
            _build_img_message("Describe", image, system_message=None)
        )
        url = next(c for c in content if c["type"] == "image_url")["image_url"]["url"]
        raw = base64.b64decode(url.removeprefix("data:image/png;base64,"))
        decoded = PIL.Image.open(io.BytesIO(raw))
        assert decoded.size == (64, 48)
