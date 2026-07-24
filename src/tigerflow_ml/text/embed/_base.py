import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams
from tigerflow_ml.utils import (
    _IMG_EXTENSIONS,
    EmptyFileError,
    load_images,
    read_text_file_strict,
)

if TYPE_CHECKING:
    from PIL import Image as PILImage


_TEXT_EXTENSIONS = [".txt", ".text", ".md", ".log", ".rtf"]
_AUDIO_EXTENSIONS = [".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"]
_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]


class _EmbedBase:
    """Embed inputs using Hugging Face sentence-transformers models."""

    class Params(HFParams):
        per_line: Annotated[
            bool,
            typer.Option(
                help="Embed each non-empty line of the input file independently, "
                "producing one vector per line instead of a single vector for the "
                "whole file."
            ),
        ] = False

        batch_size: Annotated[
            int,
            typer.Option(
                help="Number of lines encoded per batch when --per-line is set, "
                "number of pages per batch if embedding pdfs, or number of "
                "frames if embedding videos",
                min=1,
            ),
        ] = 32

        sample_fps: Annotated[
            float,
            typer.Option(
                help="Frames per second to sample from video. "
                "Set to 0 to process every frame.",
                min=0,
            ),
        ] = 1.0

        prompt: Annotated[
            str | None,
            typer.Option(
                help="Raw text prepended to each input before encoding (e.g. "
                "'query: '). Mutually exclusive with --prompt-name."
            ),
        ] = None

        prompt_name: Annotated[
            str | None,
            typer.Option(
                help="Name of a prompt predefined in the model's config (e.g. "
                "'query' or 'passage' for e5/bge models). Mutually exclusive "
                "with --prompt."
            ),
        ] = None

        normalize: Annotated[
            bool,
            typer.Option(
                help="Whether to normalize returned vectors to have length 1."
            ),
        ] = False

        truncate_dim: Annotated[
            int | None,
            typer.Option(help="The dimension to truncate sentence embeddings to."),
        ] = None

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from sentence_transformers import SentenceTransformer

        if context.prompt is not None and context.prompt_name is not None:
            raise ValueError("--prompt and --prompt-name are mutually exclusive")

        device = context.device
        if context.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(context.seed)
        logger.info(f"   Loading model {context.model}")
        try:
            context.embedder = SentenceTransformer(
                context.model,
                revision=context.revision,
                cache_folder=context.cache_dir,
                device=device,
                local_files_only=not context.allow_fetch,
            )
        except OSError as e:
            if not context.allow_fetch:
                raise RuntimeError(
                    f"'{context.model}' not found in cache ({context.cache_dir}). "
                    "Run with --allow_fetch or download manually."
                ) from e
            raise
        logger.info(
            f"   Embedding dimension: {context.embedder.get_embedding_dimension()}"
        )

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        if output_file.suffix.lower() != ".npy":
            raise ValueError(
                f"{output_file.suffix} is not a supported output for embed — "
                "save to a .npy file."
            )

        encode_kwargs = {
            "normalize_embeddings": context.normalize,
            "show_progress_bar": False,
        }
        if context.prompt is not None:
            encode_kwargs["prompt"] = context.prompt
        if context.prompt_name is not None:
            encode_kwargs["prompt_name"] = context.prompt_name
        if context.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = context.truncate_dim
        logger.info(f"   Encode kwargs: {encode_kwargs}")

        if input_file.suffix.lower() in _TEXT_EXTENSIONS:
            embeddings = _EmbedBase._embed_text(
                context=context, input_file=input_file, encode_kwargs=encode_kwargs
            )
        elif input_file.suffix.lower() in _IMG_EXTENSIONS:
            embeddings = _EmbedBase._embed_image(
                context=context, input_file=input_file, encode_kwargs=encode_kwargs
            )
        elif input_file.suffix.lower() in _AUDIO_EXTENSIONS:
            embeddings = _EmbedBase._embed_audio(
                context=context, input_file=input_file, encode_kwargs=encode_kwargs
            )
        elif input_file.suffix.lower() in _VIDEO_EXTENSIONS:
            embeddings = _EmbedBase._embed_video(
                context=context, input_file=input_file, encode_kwargs=encode_kwargs
            )
        else:
            raise ValueError(
                f"File extension {input_file.suffix} not currently supported - "
                "raise an issue on Github"
            )
        np.save(output_file, embeddings)

    @staticmethod
    def _embed_text(
        context: SetupContext, input_file: Path, encode_kwargs: dict[str, Any]
    ):
        content = read_text_file_strict(input_file)
        if context.per_line:
            texts = [line.strip() for line in content.splitlines() if line.strip()]
            embeddings = context.embedder.encode(
                texts, batch_size=context.batch_size, **encode_kwargs
            )
            logger.info(
                f"   Embedded {len(texts)} line(s) with shape {embeddings.shape}"
            )
        else:
            embeddings = context.embedder.encode_document(content, **encode_kwargs)
            logger.info(f"   Embedded 1 document with shape {embeddings.shape}")
        return embeddings

    @staticmethod
    def _embed_image(
        context: SetupContext, input_file: Path, encode_kwargs: dict[str, Any]
    ):
        images = load_images(input_file)
        if len(images) == 0:
            raise EmptyFileError(f"{input_file} is empty")
        if len(images) == 1:
            embeddings = context.embedder.encode(images[0], **encode_kwargs)
            logger.info(f"   Embedded image with shape {embeddings.shape}")
        else:  # multi-page
            embeddings = context.embedder.encode(
                images, batch_size=context.batch_size, **encode_kwargs
            )
            logger.info(
                f"   Embedded {len(images)} page(s) with shape {embeddings.shape}"
            )
        return embeddings

    @staticmethod
    def _embed_audio(
        context: SetupContext, input_file: Path, encode_kwargs: dict[str, Any]
    ):
        try:
            sampling_rate = context.embedder[
                0
            ].processor.feature_extractor.sampling_rate
        except (AttributeError, IndexError) as e:
            raise RuntimeError(
                "Could not determine the expected audio sampling rate for "
                f"'{context.model}'. This model may not expose a standard audio "
                "feature extractor (processor.feature_extractor.sampling_rate). "
            ) from e
        audio = load_audio(input_file=input_file, sampling_rate=sampling_rate)
        if audio.size == 0:
            raise EmptyFileError(f"{input_file} is empty")
        embeddings = context.embedder.encode(
            {"array": audio, "sampling_rate": sampling_rate}, **encode_kwargs
        )
        logger.info(
            f"   Embedded audio with shape {embeddings.shape} "
            f"(sampling rate {sampling_rate}Hz)"
        )
        return embeddings

    @staticmethod
    def _embed_video(
        context: SetupContext, input_file: Path, encode_kwargs: dict[str, Any]
    ):

        embeddings = []

        for batch in _batched(
            _iter_frames(input_file, context.sample_fps), context.batch_size
        ):
            images = [img for _, _, img in batch]
            logger.info(f"      Embedding {len(images)} frames...")
            embedding = context.embedder.encode(images, **encode_kwargs)
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        logger.info(
            f"   Embedded {embeddings.shape[0]} frame(s) with shape {embeddings.shape}"
        )
        return embeddings


def load_audio(input_file: Path, sampling_rate: int = 16000) -> np.ndarray:
    """Delete once PR #176 merged (will be in shared utils)"""
    import soundfile as sf
    import soxr

    array, sr = sf.read(str(input_file), dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != sampling_rate:
        array = soxr.resample(array, sr, sampling_rate)
    return np.ascontiguousarray(array, dtype=np.float32)


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
                    f"Requested sample_fps ({sample_fps}) exceeds video fps "
                    f"({video_fps:.2f}); sampling every frame."
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
