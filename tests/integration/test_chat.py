"""Integration tests for Chat task."""

import copy

import pytest

from tigerflow_ml.text.chat._base import _ChatBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="session")
def chat_dir(test_dir):
    return test_dir / "chat"


@pytest.fixture(scope="module")
def default_context(make_context):
    import gc

    import torch

    ctx = make_context(
        _ChatBase.Params,
        "chat",
    )
    _ChatBase.setup(ctx)
    yield ctx
    del ctx.LLM
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(default_context):
    assert default_context.model is not None


def test_run(
    default_context,
    chat_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(chat_dir):
        ctx = copy.copy(default_context)  # shallow clone
        output_file = make_output_path(input_file, ".txt")
        _ChatBase.run(ctx, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"chat/{input_file.stem}.defaults.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )


def test_run_params(
    default_context,
    chat_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    """
    To test all parameters that do not have a default value and are not
    applied at setup time:
        --max-image-pixels
        --video-sample-fps
    """
    for input_file in [chat_dir / "sample_image.jpg", chat_dir / "sample_video.mp4"]:
        ctx = copy.copy(default_context)  # shallow clone
        ctx.max_image_pixels = 300
        ctx.video_sample_fps = 25
        output_file = make_output_path(input_file, ".txt")
        _ChatBase.run(ctx, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"chat/{input_file.stem}.params.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )
