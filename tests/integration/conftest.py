"""Fixtures for integration tests.

Requires TIGERFLOW_ML_TEST_DIR environment variable pointing to a directory with:

    $TIGERFLOW_ML_TEST_DIR/
    ├── config.json    # optional, overrides default_config.json
    ├── ocr/
    │   ├── input/
    │   └── output/
    ├── translate/
    │   ├── input/
    │   └── output/
    ├── transcribe/
    │   ├── input/
    │   └── output/
    └── detect/
        ├── input/
        └── output/
"""

import json
import os
from pathlib import Path

import pytest
from tigerflow.utils import SetupContext

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


def pytest_collection_modifyitems(items):
    """Automatically mark tests in this directory as integration."""
    integration_dir = Path(__file__).parent
    for item in items:
        if integration_dir in item.path.parents:
            item.add_marker(pytest.mark.integration)


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
    """Factory to return all files in a task's input/ directory."""

    def _get(task_dir: Path) -> list[Path]:
        input_dir = task_dir / "input"
        assert input_dir.is_dir(), f"Missing input directory: {input_dir}"
        files = sorted(f for f in input_dir.iterdir() if f.is_file())
        assert files, f"No input files found in {input_dir}"
        return files

    return _get


@pytest.fixture(scope="session")
def make_output_path():
    """Factory to build an output path, ensuring the output directory exists."""

    def _make(task_dir: Path, input_file: Path, ext: str) -> Path:
        output_dir = task_dir / "output"
        output_dir.mkdir(exist_ok=True)
        return output_dir / (input_file.stem + ext)

    return _make


@pytest.fixture(scope="session")
def make_context(config):
    """Factory for creating a SetupContext with task params and config applied."""

    def _make(params_cls, task_name: str, **overrides):
        params = params_cls()
        ctx = SetupContext()

        # Apply param defaults
        for key, value in vars(params).items():
            if not key.startswith("_"):
                setattr(ctx, key, value)

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
