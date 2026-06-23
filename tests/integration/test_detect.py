"""Integration tests for Detect task."""

import gc

import pytest

from tigerflow_ml.image.detect._base import _DetectBase

from .conftest import assert_or_update_snapshot


@pytest.fixture(scope="module")
def default_context(make_context):
    import torch

    ctx = make_context(_DetectBase.Params, "detect")
    _DetectBase.setup(ctx)
    yield ctx
    del ctx.pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def zero_shot_context(make_context):
    import torch

    ctx = make_context(
        _DetectBase.Params,
        "detect",
        model="google/owlv2-base-patch16-ensemble",
        labels="whale,cat,dog,person,car,bird",
    )
    _DetectBase.setup(ctx)
    yield ctx
    del ctx.pipeline
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(default_context):
    assert default_context.pipeline is not None
    assert default_context.is_zero_shot is False


def test_run_image(
    default_context,
    detect_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(detect_dir):
        if input_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        output_file = make_output_path(input_file, ".json")
        _DetectBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"detect/{input_file.stem}.json",
            snapshot_dir,
            update_snapshots,
        )


def test_run_video(
    default_context,
    detect_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(detect_dir):
        if input_file.suffix.lower() not in {".mp4", ".mov", ".webm"}:
            continue
        output_file = make_output_path(input_file, ".json")
        _DetectBase.run(default_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"detect/{input_file.stem}.json",
            snapshot_dir,
            update_snapshots,
        )


def test_setup_zero_shot(zero_shot_context):
    assert zero_shot_context.pipeline is not None
    assert zero_shot_context.is_zero_shot is True
    assert zero_shot_context.labels_list == [
        "whale",
        "cat",
        "dog",
        "person",
        "car",
        "bird",
    ]


def test_run_zero_shot_image(
    zero_shot_context,
    detect_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(detect_dir):
        if input_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        output_file = make_output_path(input_file, ".json", prefix="zero-shot")
        _DetectBase.run(zero_shot_context, input_file, output_file)

        text = output_file.read_text(encoding="utf-8")
        assert_or_update_snapshot(
            text,
            f"detect/{input_file.stem}.zero-shot.json",
            snapshot_dir,
            update_snapshots,
        )


def test_zero_shot_requires_labels(make_context):
    ctx = make_context(
        _DetectBase.Params,
        "detect",
        model="google/owlv2-base-patch16-ensemble",
        labels="",
    )
    with pytest.raises(ValueError, match="requires --labels"):
        _DetectBase.setup(ctx)
