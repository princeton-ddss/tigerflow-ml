"""Integration test for OCR with:
--json-schema
--system-message
--max-model-len

"""

import json

import pytest

from tigerflow_ml.text.ocr._base import _OCRBase

from .conftest import assert_or_update_snapshot

_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "organization": {"type": "string"},
        "location": {"type": "string"},
        "title": {"type": "string"},
        "dates": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["organization", "location", "title", "dates", "bullets"],
    "additionalProperties": False,
}

RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "contact": {"type": "string"},
        "objective": {"type": "string"},
        "education": {"type": "array", "items": _ENTRY_SCHEMA},
        "work_experience": {"type": "array", "items": _ENTRY_SCHEMA},
        "volunteer_experience": {"type": "array", "items": _ENTRY_SCHEMA},
    },
    "required": [
        "name",
        "contact",
        "objective",
        "education",
        "work_experience",
        "volunteer_experience",
    ],
    "additionalProperties": False,
}


@pytest.fixture(scope="session")
def ocr_dir(test_dir):
    return test_dir / "ocr"


@pytest.fixture(scope="module")
def json_schema_context(make_context):
    import gc

    import torch

    ctx = make_context(
        _OCRBase.Params,
        "ocr",
        prompt="Extract the contents of this resume as JSON.",
        system_message="You are a helpful assistant",
        max_model_len=4000,
        json_schema=json.dumps(RESUME_SCHEMA),
    )
    _OCRBase.setup(ctx)
    yield ctx
    del ctx.LLM
    gc.collect()
    torch.cuda.empty_cache()


def test_run(
    json_schema_context,
    ocr_dir,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    input_file = ocr_dir / "resume.jpeg"
    output_file = make_output_path(input_file, ".json")
    _OCRBase.run(json_schema_context, input_file, output_file)

    text = output_file.read_text(encoding="utf-8")
    # Output must satisfy the schema's shape, not just be valid JSON.
    pages = json.loads(text)
    assert len(pages) == 1
    assert set(pages[0]) == set(RESUME_SCHEMA["required"])

    assert_or_update_snapshot(
        text,
        f"ocr/{input_file.stem}.json",
        snapshot_dir,
        update_snapshots,
        threshold=0.9,
    )
