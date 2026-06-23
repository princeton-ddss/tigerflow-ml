"""
Detect objects in images and videos using HuggingFace detection models.

Supports both fixed-class models (e.g. RT-DETR, DETR) via the object-detection
pipeline and open-vocabulary models (e.g. Grounding DINO, OWLv2) via the
zero-shot-object-detection pipeline. The pipeline type is resolved from the model.

For video input, frames are sampled at a configurable rate and processed in batches.
"""

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict

if TYPE_CHECKING:
    from PIL import Image as PILImage

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class _FramedOutput(TypedDict):
    frame: int
    timestamp: float
    detections: list[dict]


class _DetectBase:
    """Detect objects in images and videos using HuggingFace detection models."""

    class Params(HFParams):
        labels: Annotated[
            str,
            typer.Option(
                help="Comma-separated labels for zero-shot detection "
                "(e.g. 'cat,dog,person'). Required for zero-shot models, "
                "ignored for fixed-class models."
            ),
        ] = ""

        threshold: Annotated[
            float,
            typer.Option(help="Minimum confidence score for detections"),
        ] = 0.3

        batch_size: Annotated[
            int,
            typer.Option(help="Number of frames to process in parallel on GPU"),
        ] = 4

        sample_fps: Annotated[
            float,
            typer.Option(
                help="Frames per second to sample from video. "
                "Set to 0 to process every frame.",
                min=0,
            ),
        ] = 1.0

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from transformers import AutoConfig, pipeline, set_seed
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES,
        )

        set_seed(context.seed)

        logger.info("Setting up detection model...")
        logger.info("Model: {}", context.model)

        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            config = AutoConfig.from_pretrained(
                context.model,
                revision=context.revision,
                cache_dir=context.cache_dir or None,
                local_files_only=not context.allow_fetch,
            )
        except OSError as e:
            if not context.allow_fetch:
                raise RuntimeError(
                    f"'{context.model}' not found in cache ({context.cache_dir}). "
                    "Run with --allow_fetch or download manually."
                ) from e
            raise

        is_zero_shot = (
            config.model_type in MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES
        )

        if is_zero_shot and not context.labels:
            msg = (
                f"Model {context.model!r} requires --labels for zero-shot detection "
                f"(e.g. --labels 'cat,dog,person')"
            )
            raise ValueError(msg)

        pipeline_type = (
            "zero-shot-object-detection" if is_zero_shot else "object-detection"
        )
        logger.info("Pipeline: {} (model_type: {})", pipeline_type, config.model_type)

        try:
            context.pipeline = pipeline(
                pipeline_type,
                model=context.model,
                revision=context.revision,
                device=device,
                local_files_only=not context.allow_fetch,
                model_kwargs={"cache_dir": context.cache_dir or None},
            )
        except OSError as e:
            if not context.allow_fetch:
                raise RuntimeError(
                    f"'{context.model}' not found in cache ({context.cache_dir}). "
                    "Run with --allow_fetch or download manually."
                ) from e
            raise

        context.is_zero_shot = is_zero_shot
        context.labels_list = (
            [s.strip() for s in context.labels.split(",") if s.strip()]
            if context.labels
            else []
        )

        logger.info("Detection ready on device: {}", device)

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        logger.info("Processing: {}", input_file)

        is_video = input_file.suffix.lower() in _VIDEO_EXTENSIONS
        if is_video:
            output = _run_video(context, input_file)
        else:
            output = _run_image(context, input_file)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logger.info("Done")


def _format_detections(results: list[dict]) -> list[dict]:
    return [
        {
            "label": r["label"],
            "score": round(r["score"], 4),
            "box": {
                "xmin": round(r["box"]["xmin"]),
                "ymin": round(r["box"]["ymin"]),
                "xmax": round(r["box"]["xmax"]),
                "ymax": round(r["box"]["ymax"]),
            },
        }
        for r in results
    ]


def _detect_batch(context: SetupContext, images: list) -> list[list[dict]]:
    """Run detection on a batch of PIL images and return formatted results."""
    kwargs = {"threshold": context.threshold}
    if context.is_zero_shot:
        kwargs["candidate_labels"] = context.labels_list

    batch_results = context.pipeline(images, batch_size=len(images), **kwargs)
    return [_format_detections(r) for r in batch_results]


def _run_image(context: SetupContext, input_file: Path) -> list[dict]:
    """Run detection on a single image file."""
    from PIL import Image

    image = Image.open(input_file)
    if image.mode != "RGB":
        image = image.convert("RGB")

    detections = _detect_batch(context, [image])[0]
    logger.info("Found {} detection(s)", len(detections))
    return detections


def _run_video(context: SetupContext, input_file: Path) -> list[_FramedOutput]:
    """Run detection on video frames sampled at the configured FPS."""
    output: list[_FramedOutput] = []
    total_frames = 0

    for batch in _batched(
        _iter_frames(input_file, context.sample_fps), context.batch_size
    ):
        images = [img for _, _, img in batch]
        batch_detections = _detect_batch(context, images)
        for (frame_num, timestamp, _), detections in zip(batch, batch_detections):
            output.append(
                {
                    "frame": frame_num,
                    "timestamp": round(timestamp, 3),
                    "detections": detections,
                }
            )
        total_frames += len(batch)
        logger.info("Processed {} frame(s)", total_frames)

    total = sum(len(f["detections"]) for f in output)
    logger.info("Found {} detection(s) across {} frame(s)", total, len(output))
    return output


def _batched(iterable: Iterator, n: int) -> Iterator[list]:
    """Yield successive lists of up to n items from iterable."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_frames(
    video_path: Path, sample_fps: float
) -> Iterator[tuple[int, float, "PILImage.Image"]]:
    """Yield (frame_number, timestamp_seconds, PIL.Image) sampled from a video."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Could not open video: {video_path}"
        raise ValueError(msg)

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or math.isnan(video_fps):
            msg = f"Could not determine FPS for video: {video_path}"
            raise ValueError(msg)

        if sample_fps > 0:
            if sample_fps > video_fps:
                logger.warning(
                    "Requested sample_fps ({}) exceeds video fps ({:.2f}); "
                    "sampling every frame.",
                    sample_fps,
                    video_fps,
                )
            frame_interval = max(1, int(video_fps / sample_fps))
        else:
            frame_interval = 1

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                timestamp = frame_num / video_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield (frame_num, timestamp, Image.fromarray(rgb))

            frame_num += 1
    finally:
        cap.release()
