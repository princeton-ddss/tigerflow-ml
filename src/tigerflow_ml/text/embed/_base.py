from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams
from tigerflow_ml.utils import read_text_file_strict

_TEXT_EXTENSIONS = [".txt", ".text", ".md", ".log", ".rtf"]


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
                help="Number of lines encoded per batch when --per-line is set.",
                min=1,
            ),
        ] = 32

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from sentence_transformers import SentenceTransformer

        device = context.device
        if context.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

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
        if output_file.suffix.lower() != ".npy":  # TODO: support json?
            raise ValueError(
                f"{output_file.suffix} is not a supported output for embed — "
                "save to a .npy file."
            )

        if input_file.suffix.lower() in _TEXT_EXTENSIONS:
            embeddings = _EmbedBase._embed_text(context=context, input_file=input_file)
        else:
            raise ValueError(
                f"File extension {input_file.suffix} not currently supported - "
                "raise an issue on Github"
            )
        np.save(output_file, embeddings)

    @staticmethod
    def _embed_text(context: SetupContext, input_file: Path):
        content = read_text_file_strict(input_file)
        if context.per_line:
            texts = [line.strip() for line in content.splitlines() if line.strip()]
            embeddings = context.embedder.encode(texts, batch_size=context.batch_size)
            logger.info(
                f"   Embedded {len(texts)} line(s) with shape {embeddings.shape}"
            )
        else:
            embeddings = context.embedder.encode(content)
            logger.info(f"   Embedded 1 document with shape {embeddings.shape}")
        return embeddings
