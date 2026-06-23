# syntax=docker/dockerfile:1
#
# All-tasks GPU image. The torch/vllm wheels bundle their own CUDA libraries,
# so a slim base suffices; only the host driver is injected at runtime via
# `docker run --gpus all` / `apptainer run --nv`. Models are fetched at runtime
# to the HF cache at /cache.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# libsndfile1 -> soundfile; libgl1/libglib2.0-0 -> opencv-headless; ffmpeg optional.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.8 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies in a cached layer that busts only on lock/manifest changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra vllm --no-dev --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra vllm --no-dev --frozen

RUN mkdir -p /cache /data

# Run a task module directly by overriding the entrypoint, e.g.
# `--entrypoint python ... -m tigerflow_ml.audio.transcribe.local`.
ENTRYPOINT ["tigerflow"]
