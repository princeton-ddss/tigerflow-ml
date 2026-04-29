# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tigerflow-ml is an ML task library for [TigerFlow](https://github.com/princeton-ddss/tigerflow) — private cloud ML APIs on HPC infrastructure. It provides OCR, translation, transcription, and object detection tasks, each with a Local variant (for development) and a Slurm variant (for HPC cluster execution).

## Common Commands

```bash
# Install all dev dependencies
uv sync --group dev

# Run unit tests (integration tests skip automatically without env var)
uv run pytest tests

# Run a single test
uv run pytest tests/unit/test_params.py::test_ocr_defaults

# Run integration tests (requires TIGERFLOW_ML_TEST_DIR env var)
TIGERFLOW_ML_TEST_DIR=/path/to/test/data uv run pytest tests

# Lint and format
uv run pre-commit run --all-files
uv run ruff check src/
uv run ruff check src/ --fix
uv run ruff format src/

# Type checking
uvx ty check .
```

## Architecture

### Task Pattern

Every task follows a three-file pattern under `src/tigerflow_ml/`:

- **`_base.py`** — Core logic in a `_XxxBase` class with `Params` inner class (extends `HFParams`), `setup(context)`, and `run(context, input_file, output_file)` methods
- **`local.py`** — Inherits from `_XxxBase` + `LocalTask`
- **`slurm.py`** — Inherits from `_XxxBase` + `SlurmTask`

The `SetupContext` object is populated with `Params` during init, receives models/pipelines in `setup()`, and is passed to `run()`.

### Task Modules

- `text/ocr/` — Image/PDF text extraction (GOT-OCR-2.0 default)
- `text/translate/` — Document translation (MADLAD-400, MarianMT, or text-generation models)
- `audio/transcribe/` — Audio transcription (Whisper)
- `image/detect/` — Object detection in images/video (RT-DETR, DETR, Grounding DINO, OWLv2)

### Entry Points

Tasks are registered as `tigerflow.tasks` entry points in `pyproject.toml` (e.g., `ocr`, `ocr-local`). The shared parameter base class `HFParams` lives in `src/tigerflow_ml/params.py`.

## Build & CI

- Build backend: hatchling
- Package manager: uv (lockfile enforced in CI)
- Python: 3.10–3.13 (tested in matrix)
- CI runs lockfile check, pre-commit, pytest across Python versions, then builds wheel/sdist
- CD triggers on GitHub release: publishes to PyPI via trusted publishing
