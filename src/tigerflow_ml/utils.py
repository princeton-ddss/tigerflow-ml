"""Shared util across all tasks"""

import ast
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tigerflow.logconfig import logger

if TYPE_CHECKING:
    from PIL import Image
    from transformers import PretrainedConfig, PreTrainedTokenizerBase
    from vllm.sampling_params import StructuredOutputsParams  # type: ignore


class EmptyFileError(Exception):
    """Raised when an input file is empty"""

    pass


class ModelConfigParsingError(Exception):
    """Raised when errors occur while trying
    to access, read, or parse model configs"""

    pass


# Encoding fallback chain for reading files.
# Note: utf-8-sig handles both BOM and non-BOM UTF-8 files (strips BOM if present).
# latin-1 always succeeds (maps all 256 byte values), so it must be last.
# Files encoded in non-Western encodings (Shift-JIS, GB2312, etc.) will decode
# as garbage rather than failing-consider using charset-normalizer for better detection.
ENCODING_FALLBACK_CHAIN = ["utf-8-sig", "cp1252", "iso-8859-15", "latin-1"]


def read_text_file_with_fallback(path: Path) -> str:
    """
    Read a text file, trying multiple encodings until one succeeds.

    Args:
        path: Path to the file to read.

    Returns:
        The file contents as a string.

    Raises:
        RuntimeError: If the file cannot be decoded with any encoding
            (should not happen since latin-1 accepts all byte values).
    """
    last_error = None
    for i, encoding in enumerate(ENCODING_FALLBACK_CHAIN):
        try:
            with open(path, encoding=encoding) as f:
                content = f.read()
            # Warn if we fell back to latin-1 (might be decoding garbage)
            if encoding == "latin-1" and i > 0:
                logger.warning(
                    " Fell back to latin-1 encoding - content may be incorrect"
                    " if file uses a non-Western encoding (e.g., Shift-JIS, GB2312)"
                )
            return content
        except UnicodeDecodeError as e:
            last_error = e
            continue

    # This should never be reached since latin-1 accepts all byte values
    raise RuntimeError(f"Could not decode file {path}: {last_error}")


def read_text_file_strict(path: Path) -> str:
    """
    Read a text file, trying multiple encodings until one succeeds, and raise
    if file is empty or contains only white space.

    Args:
        path: Path to the file to read.

    Returns:
        The file contents as a string.

    Raises:
        RuntimeError: If the file cannot be decoded with any encoding
            (should not happen since latin-1 accepts all byte values).
        EmptyFileError: If the file contents are empty or contain only
            white space
    """
    content = read_text_file_with_fallback(path)
    if not content.strip():
        raise EmptyFileError(f"File is empty: {path}")
    return content


_IMG_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".pdf",
    ".heic",
    ".heif",
]


def load_images(path: Path, max_images: int | None = None) -> list["Image.Image"]:
    """Load images from a file. Supports image files and PDFs.

    Args:
        path: Path to the file to read.
        max_images: The maximum number of images to return if input is a PDF.
            Defaults to None (returns all).

    Returns:
        A list of PIL images of length <= max_images.

    Raises:
        ValueError: If max_images is less than 1
    """
    from PIL import Image

    if path.suffix.lower() not in _IMG_EXTENSIONS:
        raise ValueError(
            f"{path.suffix} is not a valid file type -- please choose "
            f"one of: {_IMG_EXTENSIONS}"
        )
    if path.suffix.lower() == ".heic" or path.suffix.lower() == ".heif":
        from pillow_heif import register_heif_opener

        register_heif_opener()

    if max_images is not None and max_images < 1:
        raise ValueError(f"max_images must be greater than 0. Received {max_images}")

    if path.suffix.lower() == ".pdf":
        import pymupdf

        limit = max_images if max_images is not None else float("inf")
        images = []
        with pymupdf.open(path) as doc:
            for count, page in enumerate(doc, start=1):
                if count > limit:
                    break
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(image)
        return images

    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [image]


def parse_kwargs(value: str | dict, *, name: str = "kwargs") -> dict:
    """
    Parse a string or dict into a dict, trying JSON then Python literal syntax.

    Args:
        value: A dict (returned as-is) or a string in JSON or Python dict syntax.
        name: Used in the ValueError message to identify which argument failed.

    Raises:
        ValueError: If the string cannot be parsed as a valid dict.
    """
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"--{name} is not a valid dict: {value!r}") from e


class SchemaType(str, Enum):
    """Schema type for vllm structured response"""

    JSON = "json"
    CHOICE = "choice"
    REGEX = "regex"
    GRAMMAR = "grammar"


def process_response_schema(
    schema_type: SchemaType, schema_value: str
) -> "StructuredOutputsParams":
    """
    Build a vllm StructuredOutputsParams from an explicit type and value string.

    Args:
        schema_type: One of "choice", "json", "regex", "grammar".
        schema_value: The schema value as a string. For "choice", a list;
            for "json", a JSON object; for "regex"/"grammar", a raw string.
    """
    from vllm.sampling_params import StructuredOutputsParams  # type: ignore

    if schema_type == SchemaType.CHOICE:
        try:
            value = json.loads(schema_value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(schema_value)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    "--response-schema choice value is not a valid list: "
                    f"{schema_value!r}"
                ) from e
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(
                "--response-schema choice value must be a list of strings,"
                f" got: {value!r}"
            )

        return StructuredOutputsParams(choice=value)
    elif schema_type == SchemaType.JSON:
        try:
            value = json.loads(schema_value)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--response-schema json value is not valid JSON: {schema_value!r}"
            ) from e

        return StructuredOutputsParams(json=value)
    elif schema_type == SchemaType.REGEX:
        return StructuredOutputsParams(regex=schema_value)
    elif schema_type == SchemaType.GRAMMAR:
        return StructuredOutputsParams(grammar=schema_value)


def get_model_config(
    model: str,
    allow_fetch: bool = False,
    cache_dir: str | None = None,
    revision: str = "main",
) -> "PretrainedConfig":
    """
    Load a HuggingFace model config via transformers AutoConfig.

    Args:
        model: HuggingFace model repo ID or local path.
        allow_fetch: If False, only use locally cached files (no network access).
        cache_dir: HuggingFace cache directory for model files.
        revision: Model revision (branch, tag, or commit hash).

    Returns:
        The model's PretrainedConfig (architecture, vocab size, etc.).

    Raises:
        FileNotFoundError: If the config cannot be found in the cache dir and
            --allow-fetch is False
        ModelConfigParsingError: If the config cannot be loaded or parsed, wrapping
            any underlying error.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(
            model,
            local_files_only=not allow_fetch,
            cache_dir=cache_dir,
            revision=revision,
        )
    except OSError as e:
        if not allow_fetch:
            raise FileNotFoundError(
                f"Config for '{model}' not found in cache ({cache_dir}). "
                "Run with --allow_fetch to download, or manually with: "
                f"hf download {model}"
            ) from e

        raise ModelConfigParsingError(f"Failed to load model config: {e}") from e
    except Exception as e:
        raise ModelConfigParsingError(f"Failed to load model config: {e}") from e

    return config


def get_tokenizer(
    model_name: str,
    allow_fetch: bool,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> "PreTrainedTokenizerBase":
    """Load tokenizer, downloading if needed and allowed."""
    try:
        return _load_tokenizer(model_name, cache_dir=cache_dir, revision=revision)
    except OSError:
        if not allow_fetch:
            raise RuntimeError(
                f"Tokenizer for '{model_name}' not found in cache ({cache_dir}). "
                "Run with --allow_fetch to download, or manually with: "
                f"hf download {model_name} --include 'tokenizer*'"
            )
        logger.info("Downloading tokenizer from HuggingFace Hub...")
        _download_tokenizer(model_name, cache_dir=cache_dir, revision=revision)
        return _load_tokenizer(model_name, cache_dir=cache_dir, revision=revision)


def _load_tokenizer(
    model_name: str, cache_dir: str | None = None, revision: str | None = None
) -> "PreTrainedTokenizerBase":
    """
    Load a HuggingFace tokenizer from local cache

    Returns:
        Loaded tokenizer.

    Raises:
        OSError: If tokenizer not found in cache."""

    from transformers import AutoTokenizer

    return cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, cache_dir=cache_dir, revision=revision
        ),
    )


def _download_tokenizer(
    model_name: str, cache_dir: str | None = None, revision: str | None = None
) -> None:
    """
    Download tokenizer files from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name.
        cache_dir: Optional cache directory override.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        model_name,
        allow_patterns=["tokenizer*", "special_tokens_map.json"],
        cache_dir=cache_dir,
        revision=revision,
    )


def get_model_context_window(config: "PretrainedConfig") -> int | None:
    """
    Parses a model config to identify the model's context window

    Uses the attributes: max_position_embeddings, n_positions, n_ctx, max_seq_len,
    seq_length to attempt to identify the context window.

    Args:
        config: The model's PretrainedConfig

    Returns:
        An integer representing the context window, or None if this can't be resolved
    """
    _MAX_LEN_ATTRS = (
        "max_position_embeddings",
        "n_positions",
        "n_ctx",
        "max_seq_len",
        "seq_length",
    )
    max_model_len = next(
        (
            getattr(config, a)
            for a in _MAX_LEN_ATTRS
            if getattr(config, a, None) is not None
        ),
        None,
    )

    return max_model_len


def strip_markdown_from_json(json_str: str) -> str:
    """If str starts with ```json, strip this markdown formatting from beginning and end

    Args:
        json_str: The string that should be valid json

    Returns:
        The stripped string"""
    stripped = json_str.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```")
        stripped = stripped.strip()
    return stripped
