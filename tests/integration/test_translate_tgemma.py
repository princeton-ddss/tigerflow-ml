"""Integration tests for Translate task with a TranslateGemma model.

Kept in its own module so the default model's vLLM engine is torn down
before this one starts
"""

import pytest

from tigerflow_ml.text.translate._base import _TranslateBase
from tigerflow_ml.text.translate.translator import TgemmaTranslator

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="module")
def tgemma_context(make_context):
    import gc

    import torch

    ctx = make_context(
        _TranslateBase.Params,
        "translate",
        model="Infomaniak-AI/vllm-translategemma-4b-it",
    )
    _TranslateBase.setup(ctx)
    yield ctx
    del ctx.translator
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(tgemma_context):
    assert tgemma_context.model is not None
    assert isinstance(tgemma_context.translator, TgemmaTranslator)


def test_run(
    tgemma_context,
    translate_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(translate_dir):
        output_file = make_output_path(input_file, ".txt", prefix="tgemma")
        _TranslateBase.run(tgemma_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"translate/{input_file.stem}.tgemma.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )
