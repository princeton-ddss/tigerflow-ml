"""Shared util across all tasks"""

import ast
import json
from pathlib import Path

from tigerflow.logconfig import logger


class EmptyFileError(Exception):
    """Raised when a input file is empty"""

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


def read_file_with_fallback(path: Path) -> str:
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
