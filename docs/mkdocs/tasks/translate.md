# Translation

Translate text documents using HuggingFace Seq2Seq translation models or
text-generation (causal LM) models with a translation prompt.

## Parameters

| Parameter        | Default                  | Description                                                                              |
|------------------|--------------------------|------------------------------------------------------------------------------------------|
| `--model`        | `google/madlad400-3b-mt` | HuggingFace model repo ID                                                                |
| `--revision`     | `main`                   | Model revision (branch, tag, or commit hash)                                             |
| `--cache-dir`    |                          | HuggingFace cache directory for model files                                              |
| `--device`       | `auto`                   | Device to use (`cuda`, `cpu`, or `auto`)                                                 |
| `--source-lang`  | `en`                     | Source language code (e.g. `en`, `de`, `zh`)                                             |
| `--target-lang`  | `de`                     | Target language code (e.g. `de`, `en`, `fr`)                                             |
| `--max-length`   | `512`                    | Maximum number of tokens to generate per chunk                                           |
| `--prompt`       | *(see below)*            | Prompt template for text-generation models (uses `{source_lang}`, `{target_lang}`, `{text}`) |
| `--encoding`     | `utf-8-sig`              | Input file encoding                                                                      |

!!! note

    For MADLAD-400 (default), `--source-lang` and `--target-lang` use
    [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) language codes
    (e.g. `en`, `de`, `fr`, `zh`, `ja`). For OPUS-MT models, the language pair is encoded
    in the model name and these params are ignored.

## Chunking Strategy

Input text is split into sentences and packed into chunks that fit within the model's token
limit. This preserves sentence boundaries and provides surrounding context for better
translation quality.

If a single sentence exceeds the token limit, it is split at token boundaries as a last resort.

## Output Format

Plain text, encoded as UTF-8.

## Models

Any HuggingFace [`translation`](https://huggingface.co/models?pipeline_tag=translation) (Seq2Seq) model is
supported. Causal LMs from the [`text-generation`](https://huggingface.co/models?pipeline_tag=text-generation)
pipeline can also be used with a `--prompt` template.

### Many-to-many (Seq2Seq)

| Model | Params | Languages | License |
|-------|--------|-----------|---------|
| [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt) (default) | 3B | 400+ | Apache 2.0 |
| [`google/madlad400-7b-mt`](https://huggingface.co/google/madlad400-7b-mt) | 7B | 400+ | Apache 2.0 |

### Single language pair (OPUS-MT)

For lightweight, single-pair translation. The naming convention is
`Helsinki-NLP/opus-mt-{src}-{tgt}`.

| Model | Direction | License |
|-------|-----------|---------|
| [`Helsinki-NLP/opus-mt-en-de`](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) | English → German | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-de-en`](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) | German → English | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) | English → French | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) | English → Spanish | CC-BY-4.0 |
| [`Helsinki-NLP/opus-mt-en-zh`](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) | English → Chinese | CC-BY-4.0 |

Browse all available language pairs on the [Helsinki-NLP hub page](https://huggingface.co/Helsinki-NLP).

### Text-generation (causal LM)

Any causal language model can be used for translation via the `--prompt` parameter.
The model type is auto-detected: if the model is not an encoder-decoder, it is loaded
as a `text-generation` pipeline and the prompt template is used to format each chunk.

The default prompt template is:

```text
Translate the following text from {source_lang} to {target_lang}. Output only the translation, nothing else.

{text}
```

## Examples

### Translate English to German

Uses the default MADLAD-400 model with 400+ language support.

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

### Translate English to Chinese

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: translate
        kind: local
        module: tigerflow_ml.text.translate.local
        input_ext: .txt
        output_ext: .txt
        params:
          target_lang: zh
    ```

=== "Input"

    ```text title="abstract.txt"
    Artificial intelligence is transforming academic research
    across all disciplines.
    ```

=== "Output"

    ```text title="abstract.txt"
    人工智能正在改变所有学科的学术研究。
    ```

### Use OPUS-MT for a specific language pair

For lightweight, single-pair translation without downloading a large multilingual model:

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

!!! note

    OPUS-MT models encode the language pair in the model name. The `--source-lang`
    and `--target-lang` params are ignored for these models.

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
      target_lang: zh
```
