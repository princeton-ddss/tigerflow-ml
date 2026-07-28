# Embed

Embed text using HuggingFace sentence-transformers models.

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
| `--normalize`       | `--no-normalize`   | Whether to normalize returned vectors to have length 1                                                             |
| `--truncate-dim`    |                    | The dimension to truncate sentence embeddings to                                                                   |

## Supported Input Formats

Text files (`.txt`, `.text`, `.md`, `.log`, `.rtf`)

## Output Format

NumPy binary (`.npy`).

## Models

Any HuggingFace model compatible with the [sentence-transformers](https://sbert.net) library, including plain encoder models.

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
    params:
      model: BAAI/bge-base-en-v1.5
      encode-kwargs: {"prompt":"query: "}
      cache_dir: ~/path/to/model/hub
```
