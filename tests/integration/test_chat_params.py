"""Integration test for chat with :
--response_schema
--system_message
--max_model_len

"""

import json

import pytest

from tigerflow_ml.multimodal.chat._base import _ChatBase

from .conftest import assert_or_update_snapshot

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "main_argument": {"type": "string"},
        "rights": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "main_argument",
        "rights",
    ],
    "additionalProperties": False,
}


@pytest.fixture(scope="session")
def chat_dir(test_dir):
    return test_dir / "chat"


@pytest.fixture(scope="module")
def json_schema_context(make_context):
    import gc

    import torch

    ctx = make_context(
        _ChatBase.Params,
        "chat",
        prompt="Analyze this text to extract the main argument and rights.",
        system_message="You are a helpful assistant",
        max_model_len=5000,
        response_schema=f"json={json.dumps(RESPONSE_SCHEMA)}",
    )
    _ChatBase.setup(ctx)
    yield ctx
    del ctx.LLM
    gc.collect()
    torch.cuda.empty_cache()


def test_run(
    json_schema_context,
    chat_dir,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    input_file = chat_dir / "sample_text.txt"
    output_file = make_output_path(input_file, ".json")
    _ChatBase.run(json_schema_context, input_file, output_file)

    text = output_file.read_text(encoding="utf-8")
    result = json.loads(text)
    assert set(result) == set(RESPONSE_SCHEMA["required"])

    assert_or_update_snapshot(
        text,
        f"chat/{input_file.stem}.json",
        snapshot_dir,
        update_snapshots,
        threshold=0.9,
    )
