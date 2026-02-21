# Translation

Translate text documents using HuggingFace translation models.

## Parameters

| Parameter      | Default                      | Description                                      |
|----------------|------------------------------|--------------------------------------------------|
| `--model`      | `Helsinki-NLP/opus-mt-en-de` | HuggingFace model repo ID                        |
| `--revision`   | `main`                       | Model revision (branch, tag, or commit hash)     |
| `--cache-dir`  |                              | HuggingFace cache directory for model files      |
| `--device`     | `auto`                       | Device to use (`cuda`, `cpu`, or `auto`)         |
| `--max-length` | `512`                        | Maximum number of tokens to generate per chunk   |
| `--batch-size` | `4`                          | Number of chunks to translate in parallel on GPU |
| `--encoding`   | `utf-8-sig`                  | Input file encoding                              |

!!! note

    The `--model` parameter determines the source and target languages. HuggingFace hosts
    hundreds of [OPUS-MT models](https://huggingface.co/Helsinki-NLP) covering most language
    pairs. The naming convention is `Helsinki-NLP/opus-mt-{src}-{tgt}`, e.g.
    `Helsinki-NLP/opus-mt-de-en` for German to English.

## Chunking Strategy

Input text is split into sentences and packed into chunks that fit within the model's token
limit. This preserves sentence boundaries and provides surrounding context for better
translation quality.

If a single sentence exceeds the token limit, it is split at token boundaries as a last resort.

## Output Format

Plain text, encoded as UTF-8.

## Models

Any HuggingFace [`translation`](https://huggingface.co/models?pipeline_tag=translation) model is supported. The [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) collection provides models for most language pairs.

### Common language pairs

| Model | Direction | License |
|-------|-----------|---------|
| [`Helsinki-NLP/opus-mt-en-de`](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) (default) | English → German | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-de-en`](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) | German → English | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) | English → French | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) | English → Spanish | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-zh`](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) | English → Chinese | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-zh-en`](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) | Chinese → English | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-ar`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) | English → Arabic | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-ru`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru) | English → Russian | CC-BY-4.0 |

Browse all available language pairs on the [Helsinki-NLP hub page](https://huggingface.co/Helsinki-NLP).

## Examples

### Translate English to German

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: translate
        kind: local
        module: tigerflow_ml.text.translate.local
        input_ext: .txt
        output_ext: .txt
    ```

=== "Input"

    ```text title="article.txt"
    The quick brown fox jumps over the lazy dog. This sentence
    contains every letter of the English alphabet. It has been
    used as a typing exercise for over a century.
    ```

=== "Output"

    ```text title="article.txt"
    Der schnelle braune Fuchs springt über den faulen Hund.
    Dieser Satz enthält jeden Buchstaben des englischen Alphabets.
    Er wird seit über einem Jahrhundert als Tippübung verwendet.
    ```

### Translate Spanish to English

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: translate
        kind: local
        module: tigerflow_ml.text.translate.local
        input_ext: .txt
        output_ext: .txt
        params:
          model: Helsinki-NLP/opus-mt-es-en
    ```

=== "Input"

    ```text title="documento.txt"
    La inteligencia artificial está transformando la investigación
    académica en todas las disciplinas.
    ```

=== "Output"

    ```text title="documento.txt"
    Artificial intelligence is transforming academic research
    across all disciplines.
    ```

### Run on HPC with Slurm

For bulk translation of large document collections, use the Slurm variant to distribute
work across compute nodes:

```yaml title="config.yaml"
tasks:
  - name: translate
    kind: slurm
    module: tigerflow_ml.text.translate.slurm
    input_ext: .txt
    output_ext: .txt
    max_workers: 4
    worker_resources:
      cpus: 2
      gpus: 1
      memory: 16G
      time: 02:00:00
    params:
      model: Helsinki-NLP/opus-mt-en-zh
      batch_size: 8
```
