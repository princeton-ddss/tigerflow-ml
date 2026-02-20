# Translation

Translate text documents using Hugging Face translation models.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `Helsinki-NLP/opus-mt-en-de` | HuggingFace model repo ID |
| `--revision` | `main` | Model revision (branch, tag, or commit hash) |
| `--cache-dir` | | HuggingFace cache directory for model files |
| `--device` | `auto` | Device to use (`cuda`, `cpu`, or `auto`) |
| `--max-length` | `512` | Maximum length of generated translation |
| `--batch-size` | `4` | Number of chunks to translate in parallel on GPU |
| `--encoding` | `utf-8-sig` | Input file encoding |

## Chunking Strategy

Input text is split into sentences and packed into chunks that fit within the model's token limit. This preserves sentence boundaries and provides surrounding context for better translation quality.

If a single sentence exceeds the token limit, it is split at token boundaries as a last resort.

## Output Format

Plain text, encoded as UTF-8.
