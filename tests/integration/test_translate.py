"""Integration tests for Translate task."""

import copy

import pytest

from tigerflow_ml.text.translate._base import _TranslateBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="module")
def default_context(make_context):
    import gc

    import torch

    ctx = make_context(_TranslateBase.Params, "translate")
    _TranslateBase.setup(ctx)
    yield ctx
    del ctx.translator
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(default_context):
    assert default_context.model is not None


def test_run(
    default_context,
    translate_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(translate_dir):
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"translate/{input_file.stem}.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )


def test_run_with_custom_prompt_no_lang(
    default_context,
    translate_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(translate_dir):
        ctx = copy.copy(default_context)  # shallow clone
        ctx.prompt_template = "Translate to English: {text}."
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(ctx, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"translate/{input_file.stem}.no-lang.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )


def test_run_with_custom_prompt(
    default_context,
    translate_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(translate_dir):
        ctx = copy.copy(default_context)  # shallow clone
        ctx.prompt_template = "Translate from {source_lang} to {target_lang}."
        ctx.source_lang = "it"
        ctx.target_lang = "en"
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(ctx, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"translate/{input_file.stem}.custom.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )
