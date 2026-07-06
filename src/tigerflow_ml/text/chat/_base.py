"""
Apply a chat prompt to input texts using Hugging Face models.
"""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import VLLMParams
from tigerflow_ml.utils import (
    IMG_EXTENSIONS,
    load_audio,
    load_images,
    load_video,
    parse_kwargs,
    process_response_schema,
    read_text_file_strict,
)

_TEXT_EXTENSIONS = [".txt", ".text", ".md", ".log", ".rtf"]
_AUDIO_EXTENSIONS = [".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"]
_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]


class _ChatBase:
    """Analyze text using Hugging Face models."""

    class Params(VLLMParams):
        prompt: Annotated[
            str,
            typer.Option(
                help="Prompt for text-generation models. "
                "Use {text} as a placeholder for text file contents. "
                "If '{text}' is not included, file content will follow the prompt"
            ),
        ]

        max_image_pixels: Annotated[
            int | None,
            typer.Option(
                help="Maximum image dimension in pixels (width or height). "
                "Larger images are downscaled while preserving aspect ratio."
            ),
        ] = None

        audio_sampling_rate: Annotated[
            int,
            typer.Option(
                help="Sampling rate (Hz) audio inputs are resampled to before "
                "being sent to the model."
            ),
        ] = 16000

        video_sample_fps: Annotated[
            float | None,
            typer.Option(
                help="Frame rate (fps) video inputs are resampled to before "
                "being sent to the model. Frames are dropped to hit this rate. "
                "Defaults to None (no resampling).",
            ),
        ] = None

        temperature: Annotated[
            float,
            typer.Option(
                help="The model temperature. Lower numbers make models more"
                " deterministic",
                min=0.0,
                max=2.0,
            ),
        ] = 0.0

        response_schema: Annotated[
            str | None,
            typer.Option(
                help=(
                    "Constrain the model's output format using vllm structured outputs."
                    " Format: '<type>=<value>'. "
                    "Types: "
                    'choice (list of strings, e.g. choice=["Yes","No"]), '
                    'json (JSON schema dict, e.g. json={"type":"object",...}), '
                    "regex (regular expression, e.g. regex=[0-9]+), "
                    "grammar (EBNF/GBNF grammar string)."
                )
            ),
        ] = None

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from huggingface_hub import snapshot_download

        if context.max_model_len and context.max_tokens >= context.max_model_len:
            raise ValueError(
                f"max_tokens ({context.max_tokens}) must be smaller than "
                f"max_model_len ({context.max_model_len}) — increase "
                "--max-model-len or decrease --max-tokens"
            )

        logger.info(f"  Setting up {context.model}...")

        try:
            resolved_model = snapshot_download(
                repo_id=context.model,
                cache_dir=context.cache_dir,
                local_files_only=not context.allow_fetch,
                revision=context.revision,
            )
        except OSError as e:
            if not context.allow_fetch:
                logger.error(f"Model '{context.model}' not found in cache.")
                logger.error(
                    "  Run with --allow-fetch to download, or pre-download with:"
                )
                logger.error(f"    hf download {context.model}")
                raise typer.Exit(1)
            raise RuntimeError(f"Failed to download '{context.model}': {e}") from e

        from vllm import LLM, SamplingParams  # type: ignore

        tp = torch.cuda.device_count() or 1

        user_llm_kwargs = parse_kwargs(context.llm_kwargs)
        llm_kwargs: dict[str, Any] = {
            "tensor_parallel_size": tp,
            "max_model_len": context.max_model_len,
        }
        llm_kwargs.update(user_llm_kwargs)
        logger.info(f"    llm_kwargs={llm_kwargs}")

        context.LLM = LLM(model=resolved_model, **llm_kwargs)

        user_sampling_kwargs = parse_kwargs(context.sampling_kwargs)
        sampling_kwargs: dict[str, Any] = {
            "temperature": context.temperature,
            "seed": context.seed,
            "max_tokens": context.max_tokens,
        }
        if context.response_schema is not None:
            from tigerflow_ml.utils import SchemaType

            schema_type, sep, schema_value = context.response_schema.partition("=")
            if not sep:
                raise ValueError(
                    f"--response-schema must be in the form '<type>=<value>', got: "
                    f"{context.response_schema!r}. Valid types: choice, json, regex,"
                    " grammar"
                )
            sampling_kwargs["structured_outputs"] = process_response_schema(
                SchemaType(schema_type.strip()), schema_value.strip()
            )

        sampling_kwargs.update(user_sampling_kwargs)
        logger.info(f"    sampling_kwargs={sampling_kwargs}")

        context.sampling_params = SamplingParams(**sampling_kwargs)

        user_chat_kwargs = parse_kwargs(context.chat_kwargs)
        context.chat_kwargs = {
            "sampling_params": context.sampling_params,
            "use_tqdm": False,
        }
        context.chat_kwargs.update(user_chat_kwargs)
        logger.info(f"    chat_kwargs={context.chat_kwargs}")

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):

        if input_file.suffix.lower() in _TEXT_EXTENSIONS:
            result = _ChatBase._process_text_file(context, input_file)
        elif input_file.suffix.lower() in IMG_EXTENSIONS:
            if input_file.suffix.lower() == ".pdf":
                raise ValueError(
                    f"File extension {input_file.suffix} not currently supported - "
                    "raise an issue on Github"
                )
            result = _ChatBase._process_img_file(context, input_file)
        elif input_file.suffix.lower() in _AUDIO_EXTENSIONS:
            result = _ChatBase._process_audio_file(context, input_file)
        elif input_file.suffix.lower() in _VIDEO_EXTENSIONS:
            result = _ChatBase._process_video_file(context, input_file)
        else:
            raise ValueError(
                f"File extension {input_file.suffix} not currently supported - "
                "raise an issue on Github"
            )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)

    @staticmethod
    def _process_text_file(context: SetupContext, input_file: Path) -> str:
        content = read_text_file_strict(input_file)

        message = _build_txt_message(
            prompt=context.prompt,
            text=content,
            system_message=context.system_message,
        )
        return _run_chat(context, message)

    @staticmethod
    def _process_img_file(context: SetupContext, input_file: Path) -> str:
        import PIL.Image

        image = next(load_images(path=input_file, max_images=1))

        if context.max_image_pixels is not None:
            original_size = image.size
            image.thumbnail(
                (context.max_image_pixels, context.max_image_pixels),
                PIL.Image.Resampling.LANCZOS,
            )
            if image.size != original_size:
                logger.info(
                    "  Resized image from {}x{} to {}x{}",
                    *original_size,
                    *image.size,
                )
        message = _build_img_message(
            prompt=context.prompt,
            image=image,
            system_message=context.system_message,
        )
        return _run_chat(context, message)

    @staticmethod
    def _process_audio_file(context: SetupContext, input_file: Path) -> str:
        audio_data = load_audio(input_file, sampling_rate=context.audio_sampling_rate)
        message = _build_audio_message(
            prompt=context.prompt,
            audio_data=audio_data,
            sampling_rate=context.audio_sampling_rate,
            system_message=context.system_message,
        )
        return _run_chat(context, message)

    @staticmethod
    def _process_video_file(context: SetupContext, input_file: Path) -> str:
        video_bytes = load_video(input_file, sample_fps=context.video_sample_fps)
        message = _build_video_message(
            prompt=context.prompt,
            video_bytes=video_bytes,
            system_message=context.system_message,
        )
        return _run_chat(context, message)


def _run_chat(context: SetupContext, message: Any) -> str:
    try:
        output = context.LLM.chat(message, **context.chat_kwargs)
    except ValueError as e:
        msg = str(e).lower()
        if (
            "max_model_len" in msg
            or "maximum context length" in msg
            or "too long" in msg
        ):
            if not context.max_model_len:
                err = (
                    "Input exceeds max-model-len. Change models or reduce the file size"
                )
            else:
                err = (
                    f"Input exceeds max-model-len ({context.max_model_len}). "
                    "Increase --max-model-len or reduce the file size"
                )
            raise ValueError(err) from e
        raise

    try:
        result = output[0].outputs[0]
        prompt_tokens = output[0].prompt_token_ids
        output_tokens = output[0].outputs[0].token_ids
        logger.info(
            f"  Inference successful! Prompt tokens: {len(prompt_tokens)}; "
            f"Output tokens: {len(output_tokens)}"
        )
    except IndexError:
        raise RuntimeError(
            f"{context.model} returned an empty output for the message: {message}"
        )
    if result.finish_reason == "length":
        logger.warning(
            f"  Output truncated at {context.max_tokens} tokens — increase "
            "--max-tokens and/or --max_model_len for a complete result"
        )
    elif result.finish_reason != "stop":
        raise RuntimeError(f"Unexpected finish reason: {result.finish_reason!r}")

    return result.text


def _build_txt_message(
    prompt: str, text: str, system_message: str | None
) -> list[dict[str, str]]:

    if "{text}" in prompt:
        try:
            prompt = prompt.format(text=text)
        except KeyError as e:
            raise ValueError(
                f"Prompt template contains unknown placeholder {e}. "
                "Only {text} is supported. Escape literal braces as {{ and }}."
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Invalid prompt template: {e}. Escape literal braces as {{ and }}."
            ) from e
    else:
        prompt = prompt + "\n" + text

    if system_message:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def _build_img_message(
    prompt: str, image, system_message: str | None
) -> list[dict[str, Any]]:
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    if system_message:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
    return [{"role": "user", "content": user_content}]


def _build_audio_message(
    prompt: str, audio_data: np.ndarray, sampling_rate: int, system_message: str | None
) -> list[dict[str, Any]]:
    import base64
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio_data, sampling_rate, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    user_content: list[dict[str, Any]] = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    if system_message:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
    return [{"role": "user", "content": user_content}]


def _build_video_message(
    prompt: str, video_bytes: bytes, system_message: str | None
) -> list[dict[str, Any]]:
    import base64

    b64 = base64.b64encode(video_bytes).decode("utf-8")

    user_content: list[dict[str, Any]] = [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    if system_message:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
    return [{"role": "user", "content": user_content}]
