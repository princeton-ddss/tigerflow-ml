"""Integration tests for Chat Completion task."""

import pytest

from tigerflow_ml.text.chat_completion._base import _ChatCompletionBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="session")
def chat_dir(test_dir):
    return test_dir / "chat-completion"


@pytest.fixture(scope="session")
def default_context(make_context):
    ctx = make_context(_ChatCompletionBase.Params, "chat-completion")
    _ChatCompletionBase.setup(ctx)
    return ctx


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
        _ChatCompletionBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"chat-completion/{input_file.stem}.txt",
            snapshot_dir,
            update_snapshots,
        )
