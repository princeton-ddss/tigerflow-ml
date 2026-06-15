"""Fixtures for integration tests.

Running
-------
Integration tests are gated by the ``TIGERFLOW_ML_INTEGRATION_TESTS``
environment variable. Without it, the entire suite is skipped. To run::

    TIGERFLOW_ML_INTEGRATION_TESTS=1 uv run pytest tests

Test data directory
-------------------
By default the suite reads from the committed fixtures at
``tests/integration/fixtures/``. To point at a different (e.g. larger)
fixture set on a particular machine, set ``TIGERFLOW_ML_TEST_DIR``::

    $TIGERFLOW_ML_TEST_DIR/
    ├── config.json    # optional, overrides default_config.json
    ├── ocr/           # e.g. sample.pdf, sample.png
    ├── translate/     # e.g. sample.txt
    ├── transcribe/    # e.g. sample.wav
    └── detect/        # e.g. sample.jpg

Configuration
-------------
Default model and device settings are in ``default_config.json``. To override
per-machine, place a ``config.json`` in the test data directory. Keys are merged
recursively, so you only need to specify what differs.

Snapshots
---------
Tests compare task outputs against committed snapshots in
``tests/integration/fixtures/snapshots/``. To regenerate snapshots after an
intentional behavior change, run with ``--update-snapshots``::

    TIGERFLOW_ML_INTEGRATION_TESTS=1 uv run pytest tests --update-snapshots

The updated snapshot files should be reviewed and committed.

Models should be downloaded or cached before running. On HPC, set ``cache_dir``
in your config or export ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` to point at a
shared or scratch location.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

_RUN_VAR = "TIGERFLOW_ML_INTEGRATION_TESTS"
_TEST_DIR_VAR = "TIGERFLOW_ML_TEST_DIR"
_DEFAULT_CONFIG = Path(__file__).parent / "default_config.json"
_DEFAULT_FIXTURES = Path(__file__).parent / "fixtures"
_DEFAULT_SNAPSHOTS = _DEFAULT_FIXTURES / "snapshots"


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


def assert_or_update_snapshot(
    actual: str,
    name: str,
    snapshot_dir: Path,
    update: bool,
    threshold: float | None = None,
) -> None:
    """Compare ``actual`` against the snapshot at ``snapshot_dir/name``.

    When ``update`` is True, writes ``actual`` to the snapshot path
    instead of asserting. The snapshot file is created if missing.

    When ``threshold`` is None (default), asserts byte-equality.
    When ``threshold`` is a float in [0, 1], asserts character-level
    similarity (``difflib.SequenceMatcher.ratio()``) is at least the
    threshold — useful for nondeterministic LLM outputs.
    """
    snapshot_file = snapshot_dir / name
    if update:
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file.write_text(actual, encoding="utf-8")
        return
    if not snapshot_file.exists():
        raise AssertionError(
            f"No snapshot at {snapshot_file}. "
            f"Run with --update-snapshots to create."
        )
    expected = snapshot_file.read_text(encoding="utf-8")
    if threshold is None:
        assert actual == expected, f"Snapshot mismatch for {name}"
    else:
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, expected, actual).ratio()
        assert ratio >= threshold, (
            f"Snapshot similarity {ratio:.3f} below threshold {threshold} "
            f"for {name}"
        )


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Write task outputs to fixtures/snapshots/ instead of asserting"
        " against them.",
    )


@pytest.fixture(scope="session")
def update_snapshots(request):
    return request.config.getoption("--update-snapshots")


@pytest.fixture(scope="session")
def snapshot_dir():
    """Directory for committed snapshot outputs."""
    return _DEFAULT_SNAPSHOTS


@pytest.fixture(scope="session")
def test_dir():
    """Root test data directory.

    Defaults to the committed ``tests/integration/fixtures/`` directory.
    Set ``TIGERFLOW_ML_TEST_DIR`` to point at extended fixtures on a
    machine with larger samples.
    """
    if not os.environ.get(_RUN_VAR):
        pytest.skip(f"{_RUN_VAR} not set")
    value = os.environ.get(_TEST_DIR_VAR)
    path = Path(value) if value else _DEFAULT_FIXTURES
    if not path.is_dir():
        pytest.skip(f"Test data directory not found: {path}")
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

        ctx = SetupContext()

        # Apply param defaults. Walk MRO base-first so derived-class defaults
        # overwrite base-class ones. Fields with no class-level default
        # (required params) are skipped here and must come from config or
        # overrides below.
        for cls in reversed(params_cls.__mro__):
            for key in getattr(cls, "__annotations__", {}):
                if hasattr(cls, key):
                    setattr(ctx, key, getattr(cls, key))

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

        # Validate every declared field is now set
        missing = sorted(
            {
                key
                for cls in params_cls.__mro__
                for key in getattr(cls, "__annotations__", {})
                if not hasattr(ctx, key)
            }
        )
        if missing:
            raise ValueError(
                f"Missing required params for task '{task_name}': {missing}. "
                f"Supply via config (tasks.{task_name}) or **overrides."
            )

        return ctx

    return _make
