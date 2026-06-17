"""Integration tests for OCR task."""

import pytest

from tigerflow_ml.text.ocr._base import _OCRBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="session")
def ocr_dir(test_dir):
    return test_dir / "ocr"


@pytest.fixture(scope="module")
def default_context(make_context):
    import gc

    import torch

    ctx = make_context(_OCRBase.Params, "ocr")
    _OCRBase.setup(ctx)
    yield ctx
    del ctx.LLM
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(default_context):
    assert default_context.model is not None


def test_run(
    default_context,
    ocr_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(ocr_dir):
        output_file = make_output_path(input_file, ".txt")
        _OCRBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"ocr/{input_file.stem}.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )
