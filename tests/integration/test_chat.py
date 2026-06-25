"""Integration tests for Chat task."""

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

    ctx = make_context(_ChatBase.Params, "chat")
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
        output_file = make_output_path(input_file, ".txt")
        _ChatBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"chat/{input_file.stem}.txt",
            snapshot_dir,
            update_snapshots,
            threshold=0.9,
        )
