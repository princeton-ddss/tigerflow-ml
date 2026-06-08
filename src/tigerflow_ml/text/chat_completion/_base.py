"""
Apply a chat prompt to input texts using Hugging Face models.
"""

from pathlib import Path
from typing import Annotated, Any

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import VLLMParams
from tigerflow_ml.utils import EmptyFileError, parse_kwargs, read_file_with_fallback

_TEXT_EXTENSIONS = [".txt", ".text", ".md", ".log", ".rtf"]
_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]


class _ChatCompletionBase:
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

        from vllm import LLM, SamplingParams  # type: ignore[import-unresolved]

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
            result = _ChatCompletionBase._process_text_file(context, input_file)
        elif input_file.suffix.lower() in _IMG_EXTENSIONS:
            result = _ChatCompletionBase._process_img_file(context, input_file)
        else:
            raise ValueError(
                f"File extension {input_file.suffix} not currently supported - "
                "raise an issue on Github"
            )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)

    @staticmethod
    def _process_text_file(context: SetupContext, input_file: Path) -> str:
        content = read_file_with_fallback(input_file)

        if not content.strip():
            raise EmptyFileError("Empty file")

        message = _build_txt_message(
            prompt=context.prompt,
            text=content,
            system_message=context.system_message,
        )
        return _run_chat(context, message)

    @staticmethod
    def _process_img_file(context: SetupContext, input_file: Path) -> str:
        import PIL.Image

        image = PIL.Image.open(input_file).convert("RGB")

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


def _run_chat(context: SetupContext, message: Any) -> str:
    try:
        output = context.LLM.chat(message, **context.chat_kwargs)
    except ValueError as e:
        msg = str(e).lower()
        if "max_model_len" in msg or "too long" in msg:
            raise ValueError(
                f"Input exceeds max-model-len={context.max_model_len} — "
                "increase --max-model-len or reduce the file size"
            ) from e
        raise

    try:
        result = output[0].outputs[0]
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
