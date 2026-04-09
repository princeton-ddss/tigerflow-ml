"""Fixtures for integration tests.

Requires TIGERFLOW_ML_TEST_DIR environment variable pointing to a directory with:

    $TIGERFLOW_ML_TEST_DIR/
    ├── config.json    # optional, overrides default_config.json
    ├── ocr/           # sample files directly in task dirs
    ├── translate/
    ├── transcribe/
    └── detect/
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

_TEST_DIR_VAR = "TIGERFLOW_ML_TEST_DIR"
_DEFAULT_CONFIG = Path(__file__).parent / "default_config.json"


def _load_config(test_dir: Path) -> dict:
    """Load config, merging test dir overrides on top of defaults."""
    with open(_DEFAULT_CONFIG) as f:
        config = json.load(f)

    override_path = test_dir / "config.json"
    if override_path.is_file():
        with open(override_path) as f:
            overrides = json.load(f)
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base, mutating base."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


@pytest.fixture(scope="session")
def test_dir():
    """Root test data directory from environment."""
    value = os.environ.get(_TEST_DIR_VAR)
    if not value:
        pytest.skip(f"{_TEST_DIR_VAR} not set")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{_TEST_DIR_VAR}={value} is not a directory")
    return path


@pytest.fixture(scope="session")
def config(test_dir):
    """Merged test configuration."""
    return _load_config(test_dir)


@pytest.fixture(scope="session")
def ocr_dir(test_dir):
    return test_dir / "ocr"


@pytest.fixture(scope="session")
def translate_dir(test_dir):
    return test_dir / "translate"


@pytest.fixture(scope="session")
def transcribe_dir(test_dir):
    return test_dir / "transcribe"


@pytest.fixture(scope="session")
def detect_dir(test_dir):
    return test_dir / "detect"


@pytest.fixture(scope="session")
def get_input_files():
    """Factory to return all files in a task directory."""

    def _get(task_dir: Path) -> list[Path]:
        assert task_dir.is_dir(), f"Missing task directory: {task_dir}"
        files = sorted(f for f in task_dir.iterdir() if f.is_file())
        assert files, f"No input files found in {task_dir}"
        return files

    return _get


@pytest.fixture(scope="session")
def output_dir():
    """Temporary directory for test outputs, cleaned up after the session."""
    with tempfile.TemporaryDirectory(prefix="tigerflow_ml_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def make_output_path(output_dir):
    """Factory to build an output path in the tmp directory."""

    def _make(input_file: Path, ext: str, prefix: str = "") -> Path:
        name = (
            f"{prefix}_{input_file.stem}{ext}" if prefix else f"{input_file.stem}{ext}"
        )
        return output_dir / name

    return _make


@pytest.fixture(scope="session")
def make_context(config):
    """Factory for creating a SetupContext with task params and config applied."""

    def _make(params_cls, task_name: str, **overrides):
        from tigerflow.utils import SetupContext

        params = params_cls()
        ctx = SetupContext()

        # Apply param defaults (class-level annotations, not in __dict__).
        # Walk MRO to pick up inherited fields from HFParams.
        for cls in params_cls.__mro__:
            for key in getattr(cls, "__annotations__", {}):
                if not hasattr(ctx, key):
                    setattr(ctx, key, getattr(params, key))

        # Apply global config
        if config.get("cache_dir"):
            ctx.cache_dir = config["cache_dir"]

        # Apply task-specific config
        task_config = config.get("tasks", {}).get(task_name, {})
        for key, value in task_config.items():
            setattr(ctx, key, value)

        # Apply explicit overrides
        for key, value in overrides.items():
            setattr(ctx, key, value)

        return ctx

    return _make
