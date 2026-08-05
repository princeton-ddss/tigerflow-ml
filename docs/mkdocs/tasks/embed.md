# Embed

Embed text or images using HuggingFace sentence-transformers models.

## Parameters

| Parameter          | Default            | Description                                                                                                         |
|---------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--model`           |                    | HuggingFace model repo ID                                                                                          |
| `--revision`        | `main`             | Model revision (branch, tag, or commit hash)                                                                       |
| `--cache-dir`       |                    | HuggingFace cache directory for model files                                                                        |
| `--device`          | `auto`             | Device to use (`cuda`, `cpu`, or `auto`)                                                                           |
| `--allow-fetch`     | `--no-allow-fetch` | Allow downloads from HuggingFace Hub (network access required)                                                     |
| `--seed`            | `42`               | The seed to set for more reproducible behavior                                                                     |
| `--encode-kwargs`   | `{}`               | Additional kwargs for SentenceTransformer's `encode()` (e.g. `{'prompt':'query: '}`). Supplied values override task defaults. |
| `--batch-size`      | `32`               | Number of pages per batch when embedding a multi-page PDF                                                          |
| `--normalize`       | `--no-normalize`   | Whether to normalize returned vectors to have length 1                                                             |
| `--truncate-dim`    |                    | The dimension to truncate sentence embeddings to                                                                   |
| `--use-encode-document` | `--no-use-encode-document` | Use SentenceTransformer's `encode_document()` instead of `encode()`. See [SentenceTransformer's documentation](https://sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document) for more information. |
| `--use-encode-query` | `--no-use-encode-query` | Use SentenceTransformer's `encode_query()` instead of `encode()`. See [SentenceTransformer's documentation](https://sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query) for more information. |


## Supported Input Formats

- Text files (`.txt`, `.text`, `.md`, `.log`, `.rtf`)
- Image files (`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.heic`, `.heif`)
- PDF files (`.pdf`) — each page is rendered to an image and embedded

## Output Format

NumPy binary (`.npy`).

- A single image (or single-page PDF) produces a 1-D array of shape `(dim,)`.
- A multi-page PDF produces a 2-D array of shape `(n_pages, dim)`, one row per page.

## Models

Any HuggingFace model compatible with the [sentence-transformers](https://sbert.net) library, including plain text encoder models. Embedding image or PDF input requires a multi-modal model (e.g. CLIP-style) that supports image encoding.

## Examples

### Embed a document

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: embed
        kind: local
        module: tigerflow_ml.text.embed.local
        input_ext: .txt
        output_ext: .npy
        params:
          model: sentence-transformers/all-MiniLM-L6-v2
          allow-fetch: True
    ```

=== "Input (.txt)"

    ```text title="Raven.txt"
    "The Raven" by Edgar Allan Poe
    ```

=== "Output (.npy)"

    A single vector of shape `(384,)`.

### Embed an image

Use a multi-modal (CLIP-style) model to embed images. PDFs are supported the same
way, with one row of output per page.

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: embed
        kind: local
        module: tigerflow_ml.text.embed.local
        input_ext: .jpg
        output_ext: .npy
        params:
          model: sentence-transformers/clip-ViT-B-32
          allow-fetch: True
    ```

=== "Input"

    An image file, e.g. `photo.jpg`.

=== "Output (.npy)"

    A single vector of shape `(512,)`.

### Run on HPC with Slurm

For bulk embedding across large text collections, use the Slurm variant to distribute work across compute nodes:

```yaml title="config.yaml"
tasks:
  - name: embed
    kind: slurm
    module: tigerflow_ml.text.embed.slurm
    input_ext: .txt
    output_ext: .npy
    max_workers: 4
    worker_resources:
      cpus: 2
      gpus: 1
      memory: 16G
      time: 04:00:00
    setup_commands:
      - export HF_HUB_OFFLINE=1
    params:
      model: BAAI/bge-base-en-v1.5
      encode-kwargs: {"prompt":"query: "}
      cache_dir: ~/path/to/model/hub
```
