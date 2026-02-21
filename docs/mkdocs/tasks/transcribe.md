# Transcription

Transcribe audio to text using HuggingFace Whisper models.

## Parameters

| Parameter            | Default                    | Description                                                  |
|----------------------|----------------------------|--------------------------------------------------------------|
| `--model`            | `openai/whisper-large-v3`  | HuggingFace model repo ID                                    |
| `--revision`         | `main`                     | Model revision (branch, tag, or commit hash)                  |
| `--cache-dir`        |                            | HuggingFace cache directory for model files                   |
| `--device`           | `auto`                     | Device to use (`cuda`, `cpu`, or `auto`)                      |
| `--language`         |                            | Source language code (e.g. `en`, `de`). Empty for auto-detect |
| `--output-format`    | `text`                     | Output format: `text`, `srt`, or `json`                       |
| `--batch-size`       | `16`                       | Batch size for processing audio chunks                        |
| `--chunk-length-s`   | `30.0`                     | Length of audio chunks in seconds                             |
| `--stride-length-s`  | `5.0`                      | Overlap between chunks in seconds                             |
| `--return-timestamps`| `false`                    | Return word/segment timestamps (required for SRT)             |

## Supported Input Formats

Audio files supported by the Whisper pipeline (WAV, MP3, FLAC, OGG, etc.).

## Output Format

Depends on `--output-format`:

- **text** — Plain text transcription
- **srt** — SRT subtitle format with timestamps
- **json** — Raw Whisper output with timestamps and chunk boundaries
