"""
Apply a chat prompt to input texts using Hugging Face models.
"""

from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from huggingface_hub import snapshot_download
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

from .utils import SkippedFileError, read_file_with_fallback

# TODO: test all
_TEXT_EXTENSIONS = [".txt", ".text", ".md", ".log"]
_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"]


class _ChatCompletionBase:
    """Analyze text using Hugging Face models."""

    class Params(HFParams):
        prompt: Annotated[
            str,
            typer.Option(
                help="Prompt template for text-generation models. "
                "Use {text} as a placeholder for text file contents. "
                "If '{text}' is not included, file content will follow the prompt"
            ),
        ]

        allow_fetch: Annotated[
            bool,
            typer.Option(
                help="Allow downloading from HuggingFace Hub. "
                "Do not allow if running on a compute node without network access"
            ),
        ] = False

        system_message: Annotated[
            str | None,
            typer.Option(help="System message for chat models"),
        ] = None

        max_tokens: Annotated[
            int,
            typer.Option(help="Maximum number of tokens to generate per file"),
        ] = 512

        max_model_len: Annotated[
            int,
            typer.Option(
                help="Maximum sequence length (input + output tokens) passed to vLLM. "
                "Set this for large-context models (e.g. 8192) to avoid OOM. "
                "Defaults to the model's full context window."
            ),
        ] = 32_000

        max_image_size: Annotated[
            int,
            typer.Option(
                help="Maximum image dimension in pixels (width or height). "
                "Larger images are downscaled while preserving aspect ratio."
            ),
        ] = 1024

    @staticmethod
    def setup(context: SetupContext):
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams

        if context.max_tokens >= context.max_model_len:
            raise ValueError(
                f"max_tokens ({context.max_tokens}) must be smaller than "
                f"max_model_len ({context.max_model_len}) — increase "
                "--max-model-len or decrease --max-tokens"
            )

        try:
            config = AutoConfig.from_pretrained(
                context.model,
                local_files_only=not context.allow_fetch,
                cache_dir=context.cache_dir,
                revision=context.revision,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model config: {e}")

        _MAX_LEN_ATTRS = (
            "max_position_embeddings",
            "n_positions",
            "n_ctx",
            "max_seq_len",
            "seq_length",
        )
        # Some models store the sequence length in a nested text_config
        _config_sources = [config] + (
            [config.text_config] if hasattr(config, "text_config") else []
        )
        config_max_model_len = next(
            (
                getattr(src, a)
                for src in _config_sources
                for a in _MAX_LEN_ATTRS
                if hasattr(src, a)
            ),
            None,
        )
        # logger.info(f"    config_max_model_len={config_max_model_len}")
        if config_max_model_len is not None:
            context.max_model_len = min(context.max_model_len, config_max_model_len)

        logger.info(f"  Setting up {context.model}...")
        logger.info(f"    max_model_len={context.max_model_len}")
        logger.info(f"    max_tokens={context.max_tokens}")

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

        tp = torch.cuda.device_count() or 1

        if context.device != "auto":
            context.LLM = LLM(
                model=resolved_model,
                tensor_parallel_size=tp,
                enforce_eager=True,  # TODO: expose?
                max_model_len=context.max_model_len,
                device=context.device,
            )
        else:
            context.LLM = LLM(
                model=resolved_model,
                tensor_parallel_size=tp,
                enforce_eager=True,
                max_model_len=context.max_model_len,
            )
        context.sampling_params = SamplingParams(
            temperature=0,
            seed=42,
            max_tokens=context.max_tokens,  # TODO: expose
        )

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):

        if input_file.suffix.lower() in _TEXT_EXTENSIONS:
            result = _ChatCompletionBase._process_text_file(context, input_file)
        elif input_file.suffix.lower() in _IMG_EXTENSIONS:
            result = _ChatCompletionBase._process_img_file(context, input_file)
        else:
            raise SkippedFileError(
                f"File extension {input_file.suffix} not currently supported - "
                "raise an issue on Github"
            )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)

    @staticmethod
    def _process_text_file(context: SetupContext, input_file: Path) -> str:
        content = read_file_with_fallback(input_file)

        if not content.strip():
            raise SkippedFileError("Empty file")

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
        original_size = image.size
        image.thumbnail(
            (context.max_image_size, context.max_image_size),
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
        output = context.LLM.chat(
            message,
            sampling_params=context.sampling_params,
            use_tqdm=False,
        )
    except ValueError as e:
        msg = str(e).lower()
        if "max_model_len" in msg or "too long" in msg:
            raise SkippedFileError(
                f"Input exceeds max_model_len={context.max_model_len} — "
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
        raise SkippedFileError(f"Unexpected finish reason: {result.finish_reason!r}")

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
