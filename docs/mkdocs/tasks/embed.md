# Embed

Embed text using HuggingFace sentence-transformers models.

## Parameters

| Parameter          | Default            | Description                                                                                                      |
|---------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--model`           |                    | HuggingFace model repo ID                                                                                          |
| `--revision`        | `main`             | Model revision (branch, tag, or commit hash)                                                                       |
| `--cache-dir`       |                    | HuggingFace cache directory for model files                                                                        |
| `--device`          | `auto`             | Device to use (`cuda`, `cpu`, or `auto`)                                                                            |
| `--allow-fetch`     | `--no-allow-fetch` | Allow downloads from HuggingFace Hub (network access required)                                                     |
| `--seed`            | `42`               | The seed to set for more reproducible behavior                                                                     |
| `--per-line`        | `--no-per-line`    | Embed each non-empty line of the input file independently, producing one vector per line instead of a single vector for the whole file |
| `--batch-size`      | `32`               | Number of lines encoded per batch when `--per-line` is set                                                         |
| `--prompt`          |                    | Raw text prepended to each input before encoding (e.g. `query: `). Mutually exclusive with `--prompt-name`         |
| `--prompt-name`     |                    | Name of a prompt predefined in the model's config (e.g. `query` or `passage` for e5/bge models). Mutually exclusive with `--prompt` |
| `--normalize`       | `--no-normalize`   | Whether to normalize returned vectors to have length 1                                                             |
| `--truncate-dim`    |                    | The dimension to truncate sentence embeddings to                                                                   |

## Supported Input Formats

Text files (`.txt`, `.text`, `.md`, `.log`, `.rtf`)

## Output Format

NumPy binary (`.npy`).

- By default, the whole file is embedded as one document, producing a 1-D array of shape `(dim,)`.
- With `--per-line`, each non-empty line is embedded independently, producing a 2-D array of shape `(n_lines, dim)`.

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

### Embed each line independently

Use `--per-line` to embed a corpus file with one record per line, producing one vector per line instead of a single document vector.

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
          per-line: True
          batch-size: 3
          allow-fetch: True
    ```

=== "Input (.txt)"

    ```text title="corpus.txt"
    The quick brown fox jumps over the lazy dog.
    Princeton University is in New Jersey.
    Embeddings map text to dense vectors.
    ```

=== "Output (.npy)"

    An array of shape `(3, 384)` — one row per line.


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
      per_line: True
      batch_size: 64
      cache_dir: ~/path/to/model/hub
```
