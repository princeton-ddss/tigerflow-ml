# OCR

Extract text from images and PDFs using Hugging Face image-to-text models.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `microsoft/trocr-base-printed` | HuggingFace model repo ID |
| `--revision` | `main` | Model revision (branch, tag, or commit hash) |
| `--cache-dir` | | HuggingFace cache directory for model files |
| `--device` | `auto` | Device to use (`cuda`, `cpu`, or `auto`) |
| `--max-length` | `512` | Maximum length of generated text |
| `--batch-size` | `4` | Number of images to process in parallel on GPU |

## Supported Input Formats

- Image files (PNG, JPEG, TIFF, etc.)
- PDF files (each page is rendered and processed separately)

## Output Format

Plain text. For multi-page inputs (PDFs), pages are separated by form-feed characters (`\f`).
