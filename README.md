# tigerflow-ml

[![CI](https://github.com/princeton-ddss/tigerflow-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-ddss/tigerflow-ml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tigerflow-ml)](https://pypi.org/project/tigerflow-ml/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://princeton-ddss.github.io/tigerflow-ml/)

ML tasks for [TigerFlow](https://github.com/princeton-ddss/tigerflow) — private cloud ML APIs on HPC infrastructure.

## Installation

```bash
pip install tigerflow-ml
```

If using a task that relies on `vllm` (chat completion, OCR, or translation), install with:

```bash
pip install tigerflow-ml[vllm]
```


## Tasks

| Task             | Description                           | Entry Point                       |
|------------------|---------------------------------------|-----------------------------------|
| OCR              | Extract text from images and PDFs     | `ocr` / `ocr-local`               |
| Translation      | Translate text documents              | `translate` / `translate-local`   |
| Chat Completion  | Apply a chat prompt to images or text | `chat-completion` / `chat-completion-local` |
| Transcription    | Transcribe audio to text              | `transcribe` / `transcribe-local` |
| Object Detection | Detect objects in images and videos   | `detect` / `detect-local`         |

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
python -m tigerflow_ml.text.chat_completion.slurm --help
python -m tigerflow_ml.audio.transcribe.slurm --help
python -m tigerflow_ml.image.detect.slurm --help
```

## Container

A GPU image with all tasks (including the `vllm` extra) is published to GHCR on
each release:

```bash
docker pull ghcr.io/princeton-ddss/tigerflow-ml:latest
```

The image bundles its own CUDA libraries, so only the host NVIDIA driver is
needed at runtime — pass `--gpus all` (Docker) or `--nv` (Apptainer). Models are
fetched on first use; mount a cache at `/cache` to persist them.

The entrypoint is `tigerflow`, so arguments are passed straight to the CLI:

```bash
# Run a pipeline (config file + input/output directories)
docker run --gpus all -v "$PWD/cache:/cache" -v "$PWD:/data" \
  ghcr.io/princeton-ddss/tigerflow-ml:latest \
  run /data/pipeline.yaml /data/input /data/output

# List available tasks
docker run ghcr.io/princeton-ddss/tigerflow-ml:latest tasks list

# Run a single task module directly (override the entrypoint)
docker run --gpus all -v "$PWD/cache:/cache" -v "$PWD:/data" \
  --entrypoint python ghcr.io/princeton-ddss/tigerflow-ml:latest \
  -m tigerflow_ml.audio.transcribe.local --help
```

On HPC, convert to a Singularity/Apptainer image:

```bash
apptainer build tigerflow-ml.sif docker://ghcr.io/princeton-ddss/tigerflow-ml:latest
apptainer run --nv -B ./cache:/cache tigerflow-ml.sif run pipeline.yaml input output
apptainer exec --nv tigerflow-ml.sif python -m tigerflow_ml.audio.transcribe.local --help
```

## Development

```bash
uv sync --group dev
uv run pre-commit run --all-files
uv run pytest tests
```
