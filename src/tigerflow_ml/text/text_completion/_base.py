"""
Apply a chat prompt to input texts using Hugging Face models.
"""

from pathlib import Path
from typing import Annotated, Any, cast

import torch
import typer
from huggingface_hub import snapshot_download
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext
from vllm import LLM, SamplingParams

from tigerflow_ml.params import HFParams

from .utils import SkippedFileError, read_file_with_fallback


class _TextCompletionBase:
    """Analyze text using Hugging Face models."""

    class Params(HFParams):
        prompt_template: Annotated[
            str,
            typer.Option(
                help="Prompt template for text-generation models. "
                "Use {text} as a placeholder for the text file contents."
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

    @staticmethod
    def setup(context: SetupContext):

        # try:
        #     config = AutoConfig.from_pretrained(
        #         context.model,
        #         local_files_only=not context.allow_fetch,
        #         cache_dir=context.cache_dir,
        #         revision=context.revision,
        #     )
        # except Exception as e:
        #     raise ConfigParsingError(f"Failed to load model config: {e}")

        logger.info(f"  Setting up {context.model}...")

        if context.cache_dir is not None:
            resolved_model = snapshot_download(
                repo_id=context.model,
                cache_dir=context.cache_dir,
                local_files_only=not context.allow_fetch,
                revision=context.revision,
            )
        else:
            resolved_model = context.model

        tp = torch.cuda.device_count() or 1

        if context.device != "auto":
            context.LLM = LLM(
                model=resolved_model,
                tensor_parallel_size=tp,
                enforce_eager=True,
                device=context.device,
            )
        else:
            context.LLM = LLM(
                model=resolved_model,
                tensor_parallel_size=tp,
                enforce_eager=True,
            )
        context.sampling_params = SamplingParams(
            temperature=0, seed=42, max_tokens=context.max_tokens
        )

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):

        content = read_file_with_fallback(input_file)

        if not content.strip():
            raise SkippedFileError("Empty file")

        message = _build_message(
            prompt_template=context.prompt_template,
            text=content,
            system_message=context.system_message,
        )
        output = context.LLM.chat(
            cast(Any, message), sampling_params=context.sampling_params, use_tqdm=False
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output[0].outputs[0].text)


def _build_message(
    prompt_template: str, text: str, system_message: str | None
) -> list[dict[str, str]]:

    prompt = prompt_template.format(text=text)

    if system_message:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]
