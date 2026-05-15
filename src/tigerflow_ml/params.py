"""Shared parameter definitions for HuggingFace-based tasks."""

from typing import Annotated

import typer


class HFParams:
    """Common parameters for all HuggingFace pipeline tasks."""

    model: Annotated[
        str,
        typer.Option(help="HuggingFace model repo ID"),
    ] = ""

    revision: Annotated[
        str,
        typer.Option(help="Model revision (branch, tag, or commit hash)"),
    ] = "main"

    cache_dir: Annotated[
        str,
        typer.Option(help="HuggingFace cache directory for model files"),
    ] = ""

    device: Annotated[
        str,
        typer.Option(help="Device to use (cuda, cpu, or auto)"),
    ] = "auto"


class VLLMParams:
    """Commonb parameters for all vLLM tasks"""

    model: Annotated[
        str,
        typer.Option(help="HuggingFace model repo ID"),
    ]

    revision: Annotated[
        str,
        typer.Option(help="Model revision (branch, tag, or commit hash)"),
    ] = "main"

    cache_dir: Annotated[
        str,
        typer.Option(help="HuggingFace cache directory for model files"),
    ] = ""

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
        int | None,
        typer.Option(
            help="Maximum sequence length (input + output tokens) passed to vLLM. "
            "Set this for large-context models to avoid OOM. "
        ),
    ] = None

    temperature: Annotated[
        float,
        typer.Option(
            help="The model temperature. Lower numbers make models more deterministic",
            min=0.0,
            max=2.0,
        ),
    ] = 0.0

    seed: Annotated[
        int, typer.Option(help="The seed to set for more reproducible behavior")
    ] = 42

    llm_kwargs: Annotated[
        str,
        typer.Option(
            help="Additional kwargs for vLLM's LLM() constructor. "
            "Supplied values override task defaults."
        ),
    ] = "{}"

    sampling_kwargs: Annotated[
        str,
        typer.Option(
            help="Additional kwargs for vLLM's SamplingParams() constructor. "
            "Supplied values override task defaults."
        ),
    ] = "{}"

    chat_kwargs: Annotated[
        str,
        typer.Option(
            help="Additional kwargs for vLLM's LLM.chat(). "
            "Supplied values override task defaults."
        ),
    ] = "{}"
