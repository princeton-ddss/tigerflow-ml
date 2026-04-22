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
        str | None,
        typer.Option(help="HuggingFace cache directory for model files"),
    ] = None

    device: Annotated[
        str,
        typer.Option(help="Device to use (cuda, cpu, or auto)"),
    ] = "auto"
