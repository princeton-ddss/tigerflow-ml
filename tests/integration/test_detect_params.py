"""Integration test for Detect with --compile."""

import gc

import pytest

from tigerflow_ml.image.detect._base import _DetectBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="module")
def context(make_context):
    import torch

    ctx = make_context(_DetectBase.Params, "detect", compile=True)
    _DetectBase.setup(ctx)
    yield ctx
    del ctx.pipeline
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(context):
    assert context.pipeline is not None


def test_run(
    context,
    detect_dir,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    input_file = detect_dir / "orcas.jpg"
    output_file = make_output_path(input_file, ".json")
    _DetectBase.run(context, input_file, output_file)

    text = output_file.read_text(encoding="utf-8")
    assert_or_update_snapshot(
        text,
        f"detect/{input_file.stem}.compiled.json",
        snapshot_dir,
        update_snapshots,
        threshold=0.9,
    )
