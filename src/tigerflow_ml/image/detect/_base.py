"""
Detect objects in images and videos using HuggingFace detection models.

Supports both fixed-class models (e.g. RT-DETR, DETR) via the object-detection
pipeline and open-vocabulary models (e.g. Grounding DINO, OWLv2) via the
zero-shot-object-detection pipeline. The pipeline type is resolved from the model.

For video input, frames are sampled at a configurable rate and processed in batches.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from PIL import Image as PILImage

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

_ZERO_SHOT_MODEL_TYPES = {"grounding-dino", "owlv2", "owlvit"}

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class _DetectBase:
    """Detect objects in images and videos using HuggingFace detection models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "PekingU/rtdetr_r50vd"

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
                "Set to 0 to process every frame."
            ),
        ] = 1.0

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from transformers import AutoConfig, pipeline

        logger.info("Setting up detection model...")
        logger.info("Model: {}", context.model)

        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = AutoConfig.from_pretrained(
            context.model,
            revision=context.revision,
            cache_dir=context.cache_dir or None,
        )
        is_zero_shot = config.model_type in _ZERO_SHOT_MODEL_TYPES

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

        context.pipeline = pipeline(
            pipeline_type,
            model=context.model,
            revision=context.revision,
            device=device,
            model_kwargs={"cache_dir": context.cache_dir or None},
        )
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


def _detect(context: SetupContext, image) -> list[dict]:
    """Run detection on a single PIL image and return formatted results."""
    kwargs = {"threshold": context.threshold}
    if context.is_zero_shot:
        kwargs["candidate_labels"] = context.labels_list

    results = context.pipeline(image, **kwargs)

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


def _run_image(context: SetupContext, input_file: Path) -> list[dict]:
    """Run detection on a single image file."""
    from PIL import Image

    image = Image.open(input_file)
    if image.mode != "RGB":
        image = image.convert("RGB")

    detections = _detect(context, image)
    logger.info("Found {} detection(s)", len(detections))
    return detections


def _run_video(context: SetupContext, input_file: Path) -> list[dict]:
    """Run detection on video frames sampled at the configured FPS."""
    frames = _extract_frames(input_file, context.sample_fps)
    logger.info(
        "Extracted {} frame(s) at {} fps", len(frames), context.sample_fps or "all"
    )

    output = []
    for i in range(0, len(frames), context.batch_size):
        batch = frames[i : i + context.batch_size]
        for frame_num, timestamp, image in batch:
            detections = _detect(context, image)
            output.append(
                {
                    "frame": frame_num,
                    "timestamp": round(timestamp, 3),
                    "detections": detections,
                }
            )
        logger.info(
            "Processed frames {}-{} / {}",
            i,
            min(i + context.batch_size, len(frames)) - 1,
            len(frames),
        )

    total = sum(len(f["detections"]) for f in output)
    logger.info("Found {} detection(s) across {} frame(s)", total, len(output))
    return output


def _extract_frames(
    video_path: Path, sample_fps: float
) -> list[tuple[int, float, "PILImage.Image"]]:
    """Extract frames from a video file at the given sample rate.

    Returns a list of (frame_number, timestamp_seconds, PIL.Image) tuples.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Could not open video: {video_path}"
        raise ValueError(msg)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / sample_fps) if sample_fps > 0 else 1

    frames = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            timestamp = frame_num / video_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            frames.append((frame_num, timestamp, image))

        frame_num += 1

    cap.release()
    return frames
