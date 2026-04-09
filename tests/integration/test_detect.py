"""Integration tests for Detect task."""

import json

import pytest

from tigerflow_ml.image.detect._base import _DetectBase

_context = None


@pytest.mark.dependency()
def test_setup(make_context):
    global _context
    _context = make_context(_DetectBase.Params, "detect")
    _DetectBase.setup(_context)


@pytest.mark.dependency(depends=["test_setup"])
def test_run(detect_dir, get_input_files, make_output_path):
    for input_file in get_input_files(detect_dir):
        output_file = make_output_path(input_file, ".json")
        _DetectBase.run(_context, input_file, output_file)

        assert output_file.exists(), f"No output for {input_file.name}"
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert isinstance(data, list), f"Expected list for {input_file.name}"

        for detection in data:
            assert "label" in detection
            assert "score" in detection
            assert "box" in detection
            assert isinstance(detection["score"], float)
            box = detection["box"]
            for key in ("xmin", "ymin", "xmax", "ymax"):
                assert key in box
