# tigerflow-ml

ML tasks for TigerFlow (translation, OCR).

## Installation

```bash
pip install tigerflow-ml
```

## Tasks

- **translate**: Translate text documents using Hugging Face text-to-text models
- **ocr**: Extract text from images using Hugging Face image-to-text models

## Usage

After installation, tasks are automatically discoverable via:

```bash
tigerflow tasks list
```

Run a task directly:

```bash
python -m tigerflow_ml.translate --help
python -m tigerflow_ml.ocr --help
```
