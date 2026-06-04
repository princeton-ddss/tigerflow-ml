# Translation

Translate text documents using HuggingFace [vLLM optimized TranslateGemma](https://docs.vllm.ai/projects/recipes/en/latest/Google/TranslateGemma.html) models or chat models via vLLM.

## Parameters

| Parameter            | Default                          | Description                                                                                                                      |
|----------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `--model`            |                                  | HuggingFace model repo ID                                                                                                        |
| `--revision`         | `main`                           | Model revision (branch, tag, or commit hash)                                                                                     |
| `--cache-dir`        |                                  | HuggingFace cache directory for model files (equivalent to `HF_HOME`)                                                            |
| `--allow-fetch`      | `--no-allow-fetch`               | Allow downloads from HuggingFace Hub (network access required)                                                                   |
| `--source-lang`      |                                  | Source language code (e.g. `en`, `de`, `zh`). Overrides auto-detection when `--auto-lang-detect` is on.                          |
| `--target-lang`      | `en`                             | Target language code (e.g. `de`, `en`, `fr`)                                                                                     |
| `--auto-lang-detect` | `True`                           | Auto-detect source language with `langdetect`. When disabled (`--no-auto-lang-detect`), `--source-lang` is required.             |
| `--chunk-size`       | `900`                            | Maximum input tokens per chunk                                                                                                   |
| `--max-model-len`    | `chunk_size × 2.5 + 512`         | Maximum sequence length (input + output tokens) passed to vLLM. Capped by the model's configured context window.                 |
| `--model-backend`    | `auto`                           | Model backend: `chat`, `tgemma`, or `auto` (auto-detects from model name)                                                        |
| `--prompt-template`  | *(see below)*                    | Prompt template for chat-based models (uses `{source_lang}`, `{target_lang}`, and `{text}`). Ignored for tgemma models.          |
| `--system-message`   |                                  | System message for chat models. Ignored for tgemma models.                                                                       |
| `--temperature`      | `0.0`                            | Sampling temperature. Lower values make output more deterministic.                                                               |
| `--seed`             | `42`                             | Random seed for reproducibility                                                                                                  |
| `--llm-kwargs`       | `{}`                             | JSON string of extra kwargs for vLLM's `LLM()` constructor. Overrides task defaults.                                             |
| `--sampling-kwargs`  | `{}`                             | JSON string of extra kwargs for vLLM's `SamplingParams()`. Overrides task defaults.                                              |
| `--chat-kwargs`      | `{}`                             | JSON string of extra kwargs for `LLM.chat()`. Overrides task defaults.                                                           |


## Chunking Strategy

For long documents, input text is split into chunks based on a `chunk size`, to fit within a model's token limit. An attempt will be made to split the text at the paragraph level; for long paragraphs this may not be possible, in which case the text will be split at the sentence level. This preserves sentence boundaries and provides surrounding context for better translation quality. If a single sentence exceeds the token limit, it is split at token boundaries as a last resort.

## Output Format

Plain text, encoded as UTF-8.

## Models

All models are served via vLLM. There are two backends, selected automatically from the model name (`--model-backend auto`) or set explicitly:

- **`tgemma`** — [vLLM compatible TranslateGemma](https://docs.vllm.ai/projects/recipes/en/latest/Google/TranslateGemma.html) models (any `translategemma-*-it` repo). Uses the native `<<<source>>>{src_lang}<<<target>>>{tgt_lang}<<<text>>>{text}` prompt format and ignores `--prompt-template` and `--system-message`.
- **`chat`** — Any instruction-tuned LLM. Uses `LLM.chat()` with the configurable `--prompt-template` and `--system-message`.

The default prompt template for chat models is:

```text
Translate the following text from {source_lang} to {target_lang}. Output only the translated text, nothing else. Text: {text}
```

We recommend running this task on a sufficiently large GPU. This might mean running on an HPC compute node without network access. If this is the case for you, you'll have to download the model files before running this task. To download a gated model, first run `hf auth login` or `export HF_TOKEN="hf_token"`. Then, download the model to a cache directory of your choice (here `./.hf`):

```bash
HF_HOME=./.hf hf download Infomaniak-AI/vllm-translategemma-27b-it
```

## Examples


### Local translation with pre-downloaded TranslateGemma model

Uses a TranslateGemma model to translate a short English text to German locally with a pre-downloaded model.
Run with the command: `tigerflow run config.yaml ./input/ ./output/`.*

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: translate
        kind: local
        module: tigerflow_ml.text.translate.local
        input_ext: .txt
        output_ext: .txt
        params:
          model: Infomaniak-AI/vllm-translategemma-27b-it
          source-lang: en
          target-lang: de
          cache-dir: /path/to/.hf/hub/
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
    Dieser Satz enthält alle Buchstaben des englischen Alphabets.
    Er wird seit über einem Jahrhundert als Übung für das Tippen verwendet.
    ```

*This task can also be run from the command line with the command: `python -m tigerflow_ml.text.translate.local --input-dir ./input/ --input-ext .txt --output-dir ./output/ --output-ext .txt --model Infomaniak-AI/vllm-translategemma-27b-it --cache-dir /path/to/.hf/hub/ --source-lang en --target-lang de`


### Local translation fetching an LLM from HuggingFace Hub

Uses `google/gemma-3-4b-it` (an instruction-tuned chat model) to translate a short Chinese text to English locally with network access.

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: translate
        kind: local
        module: tigerflow_ml.text.translate.local
        input_ext: .txt
        output_ext: .txt
        params:
          source-lang: zh
          model: google/gemma-3-4b-it
          allow-fetch: True
    ```

=== "Input"

    ```text title="abstract.txt"
    人工智能正在改变所有学科的学术研究。
    ```

=== "Output"

  ```text title="abstract.txt"
  Artificial intelligence is changing academic research in all disciplines.
  ```



### Run on HPC with Slurm

For bulk translation of large document collections, use the Slurm variant to distribute work across compute nodes. This example uses a pre-downloaded `google/translategemma-27b-it` model stored in the cache directory `.hf`.

=== "Config"

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
          sbatch_options:
            - "--constraint=gpu80"
        setup_commands:
          - source .venv/bin/activate
          - export VLLM_USE_FLASHINFER_SAMPLER=0
        params:
          model: google/translategemma-27b-it
          cache-dir: /path/to/.hf/hub/
    ```
