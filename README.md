# tigerflow-ml

[![CI](https://github.com/princeton-ddss/tigerflow-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-ddss/tigerflow-ml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tigerflow-ml)](https://pypi.org/project/tigerflow-ml/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://princeton-ddss.github.io/tigerflow-ml/)

ML tasks for [TigerFlow](https://github.com/princeton-ddss/tigerflow) — private cloud ML APIs on HPC infrastructure.

## Installation

```bash
pip install tigerflow-ml
```

## Tasks

| Task             | Description                         | Entry Point                       |
|------------------|-------------------------------------|-----------------------------------|
| OCR              | Extract text from images and PDFs   | `ocr` / `ocr-local`               |
| Translation      | Translate text documents            | `translate` / `translate-local`   |
| Transcription    | Transcribe audio to text            | `transcribe` / `transcribe-local` |
| Object Detection | Detect objects in images and videos | `detect` / `detect-local`         |

Each task provides both a Slurm variant (for HPC) and a Local variant (for development).

## Usage

After installation, tasks are automatically discoverable via:

```bash
tigerflow tasks list
```

Run a task directly:

```bash
python -m tigerflow_ml.text.ocr.slurm --help
python -m tigerflow_ml.text.translate.slurm --help
python -m tigerflow_ml.audio.transcribe.slurm --help
python -m tigerflow_ml.image.detect.slurm --help
```

## Development

```bash
uv sync --group dev
uv run pre-commit run --all-files
uv run pytest tests
```
