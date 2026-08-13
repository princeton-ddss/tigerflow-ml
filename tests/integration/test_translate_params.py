"""Integration test for Translate with:
--max-model-len.
"""

import pytest

from tigerflow_ml.text.translate._base import _TranslateBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="module")
def context(make_context):
    import gc

    import torch

    ctx = make_context(_TranslateBase.Params, "translate", max_model_len=5000)
    _TranslateBase.setup(ctx)
    yield ctx
    del ctx.translator
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(context):
    assert context.model is not None


def test_run(
    context,
    translate_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(translate_dir):
        output_file = make_output_path(input_file, ".txt")
        _TranslateBase.run(context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"translate/{input_file.stem}.params.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )
