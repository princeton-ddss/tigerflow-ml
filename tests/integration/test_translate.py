"""Integration tests for Translate task."""

import copy

import pytest

from tigerflow_ml.text.translate._base import _TranslateBase


@pytest.fixture(scope="session")
def default_context(make_context):
    ctx = make_context(_TranslateBase.Params, "translate")
    _TranslateBase.setup(ctx)
    return ctx


def test_setup(default_context):
    assert default_context.model is not None


def test_run(default_context, translate_dir, get_input_files, make_output_path):
    for input_file in get_input_files(translate_dir):
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(default_context, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"


def test_run_with_custom_prompt_no_lang(
    default_context, translate_dir, get_input_files, make_output_path
):
    for input_file in get_input_files(translate_dir):
        ctx = copy.copy(default_context)  # shallow clone
        ctx.prompt_template ="Translate to English: {text}."
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(ctx, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"


def test_run_with_custom_prompt(
    default_context, translate_dir, get_input_files, make_output_path
):
    for input_file in get_input_files(translate_dir):
        ctx = copy.copy(default_context)  # shallow clone
        ctx.prompt_template ="Translate from {source_lang} to {target_lang}."
        ctx.source_lang = "it"
        ctx.target_lang = "en"
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(ctx, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        text = output_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 0, f"Empty output for {input_file.name}"
