from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams
from tigerflow_ml.utils import TEXT_EXTENSIONS, parse_kwargs, read_text_file_strict


class _EmbedBase:
    """Embed inputs using Hugging Face sentence-transformers models."""

    class Params(HFParams):
        use_encode_document: Annotated[
            bool,
            typer.Option(
                help="Whether to use SentenceTransformer's encode_document() "
                "method instead of the regular encode()."
            ),
        ] = False

        use_encode_query: Annotated[
            bool,
            typer.Option(
                help="Whether to use SentenceTransformer's encode_query() "
                "method instead of the regular encode()."
            ),
        ] = False

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

        encode_kwargs: Annotated[
            str,
            typer.Option(
                help="Additional kwargs for SentenceTransformer's encode() "
                "(e.g., {'prompt':'query: '}). Supplied values override task defaults."
            ),
        ] = "{}"

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from sentence_transformers import SentenceTransformer

        if context.use_encode_document and context.use_encode_query:
            raise ValueError(
                "--use-encode-document and --use-encode-query are mutually exclusive"
            )

        device = context.device
        if context.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(context.seed)

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

        user_encode_kwargs = parse_kwargs(context.encode_kwargs)
        context.encode_kwargs = {
            "normalize_embeddings": context.normalize,
            "show_progress_bar": False,
        }
        if context.truncate_dim is not None:
            context.encode_kwargs["truncate_dim"] = context.truncate_dim
        context.encode_kwargs.update(user_encode_kwargs)
        logger.info(f"   encode_kwargs={context.encode_kwargs}")

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        if output_file.suffix.lower() != ".npy":
            raise ValueError(
                f"{output_file.suffix} is not a supported output for embed — "
                "save to a .npy file."
            )

        if input_file.suffix.lower() in TEXT_EXTENSIONS:
            embeddings = _EmbedBase._embed_text(
                context=context,
                input_file=input_file,
            )
        else:
            raise ValueError(
                f"File extension {input_file.suffix} not currently supported - "
                "raise an issue on Github"
            )
        np.save(output_file, embeddings)

    @staticmethod
    def _embed_text(context: SetupContext, input_file: Path):
        content = read_text_file_strict(input_file)

        if context.use_encode_document:
            embeddings = context.embedder.encode_document(
                content, **context.encode_kwargs
            )
        elif context.use_encode_query:
            embeddings = context.embedder.encode_query(content, **context.encode_kwargs)
        else:
            embeddings = context.embedder.encode(content, **context.encode_kwargs)
        logger.info(f"   Embedded 1 document with shape {embeddings.shape}")
        return embeddings
