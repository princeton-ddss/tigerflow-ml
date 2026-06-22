"""
Perform OCR on images using Hugging Face image-text-to-text models.

Supports VLMs compatible with the image-text-to-text pipeline.
"""

import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import VLLMParams
from tigerflow_ml.utils import load_images, parse_kwargs

if TYPE_CHECKING:
    from PIL import Image


class OutputFormat(str, Enum):
    """Output format for OCR results."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


class _OCRBase:
    """Extract text from images using image-text-to-text models."""

    class Params(VLLMParams):
        prompt: Annotated[
            str,
            typer.Option(help="Prompt for image-text-to-text models"),
        ]

        # overrides default
        max_tokens: Annotated[
            int,
            typer.Option(help="Maximum number of tokens to generate per image"),
        ] = 4096

        # could also infer from output_file extension
        format: Annotated[
            OutputFormat,
            typer.Option(help="Validates output format: 'text', 'markdown', or 'json'"),
        ] = OutputFormat.TEXT

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams  # type: ignore[import-unresolved]

        logger.info("Setting up OCR model...")
        logger.info(f"   Model: {context.model}")

        try:
            resolved_model = snapshot_download(
                repo_id=context.model,
                cache_dir=context.cache_dir,
                local_files_only=not context.allow_fetch,
                revision=context.revision,
            )
        except OSError as e:
            if not context.allow_fetch:
                raise RuntimeError(
                    f"'{context.model}' not found in cache ({context.cache_dir}). "
                    "Run with --allow_fetch or download manually."
                ) from e
            raise

        tp = torch.cuda.device_count() or 1

        user_llm_kwargs = parse_kwargs(context.llm_kwargs)
        llm_kwargs: dict[str, Any] = {
            "tensor_parallel_size": tp,
            "use_tqdm_on_load": False,
        }
        if context.max_model_len is not None:
            llm_kwargs["max_model_len"] = context.max_model_len
        llm_kwargs.update(user_llm_kwargs)
        logger.info(f"   llm_kwargs={llm_kwargs}")

        context.LLM = LLM(model=resolved_model, **llm_kwargs)

        user_sampling_kwargs = parse_kwargs(context.sampling_kwargs)
        sampling_kwargs: dict[str, Any] = {
            "temperature": context.temperature,
            "seed": context.seed,
            "max_tokens": context.max_tokens,
        }
        sampling_kwargs.update(user_sampling_kwargs)
        logger.info(f"   sampling_kwargs={sampling_kwargs}")

        context.sampling_params = SamplingParams(**sampling_kwargs)

        user_chat_kwargs = parse_kwargs(context.chat_kwargs)
        context.chat_kwargs = {
            "sampling_params": context.sampling_params,
            "use_tqdm": False,
        }
        context.chat_kwargs.update(user_chat_kwargs)
        logger.info(f"   chat_kwargs={context.chat_kwargs}")

        logger.info("Setup complete!")

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        images = load_images(input_file)
        logger.info(f"Loaded {len(images)} image(s)")

        messages = []
        for image in images:
            message = _format_message(
                image=image,
                prompt=context.prompt,
                system_message=context.system_message,
            )
            messages.append(message)

        output = context.LLM.chat(messages, **context.chat_kwargs)
        completions = [o.outputs[0] for o in output]
        for page, completion in enumerate(completions, start=1):
            if completion.finish_reason == "length":
                if page > 1:
                    msg = (
                        f"  Output truncated at {context.max_tokens} tokens (page"
                        f"{page}) — increase --max-tokens and/or --max_model_len "
                        "for a complete result"
                    )
                else:
                    msg = (
                        f"  Output truncated at {context.max_tokens} tokens — "
                        "increase --max-tokens and/or --max_model_len for a "
                        "complete result"
                    )
                logger.warning(msg)
            elif completion.finish_reason != "stop":
                if page > 1:
                    msg = (
                        "Unexpected finish reason on page "
                        f"{page}: {completion.finish_reason!r}"
                    )
                else:
                    msg = f"Unexpected finish reason: {completion.finish_reason!r}"
                raise RuntimeError(msg)
            _validate_output(completion.text, context.format)

        output_text = "\f".join(c.text for c in completions)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)


def _format_message(
    image: "Image.Image", prompt: str, system_message: str | None = None
) -> list:
    if system_message:
        return [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _validate_output(output: str, output_format: OutputFormat):
    """Raise error if model output doesn't match specified output format"""
    if output_format == OutputFormat.TEXT:
        return
    elif output_format == OutputFormat.MARKDOWN:
        pass
    elif output_format == OutputFormat.JSON:
        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(  # including {output} may be too long
                f"Model did not return a valid json output. Returned: {output}."
                f"Try refining your prompt or use a different --format"
            ) from e

    raise ValueError(f" Unsupported output format: {output_format}")
