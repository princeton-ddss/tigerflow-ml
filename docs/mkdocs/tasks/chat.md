# Chat

Analyze text or image files using vLLM compatible HuggingFace chat models.

## Parameters

| Parameter             | Default                   | Description                                                    |
|-----------------------|---------------------------|----------------------------------------------------------------|
| `--model`             |                           | HuggingFace model repo ID                                      |
| `--prompt`            |                           | Model prompt. If using text inputs, use `{text}` as a placeholder for file contents (if not included, contents will follow the prompt) |
| `--system-message`    |                           | System message for chats                                       |
| `--max-tokens`        | `512`                     | Maximum number of tokens to generate per file                  |
| `--revision`          | `main`                    | Model revision (branch, tag, or commit hash)                   |
| `--cache-dir`         |                           | HuggingFace cache directory for model files                    |
| `--allow-fetch`       | `--no-allow-fetch`        | Allow downloads from HuggingFace Hub (network access required) |
| `--max-model-len`     |                           | Maximum sequence length (input + output tokens) passed to vLLM. Set this for large-context models to avoid OOM. |
| `--max-image-pixels`  |                           | Maximum image dimension in pixels (width or height). Larger images are downscaled while preserving aspect ratio. |
| `--temperature`       | `0`                       | The model temperature. Lower numbers make models more deterministic |
| `--response-schema`   |                           | Constrain the model's output format using vllm structured outputs. Format: `<type>=<value>`. Types: `choice` (list of strings), `json` (JSON schema dict), `regex` (regular expression), `grammar` (EBNF/GBNF grammar string). |
| `--seed`              | `42`                      | The seed to set for more reproducible behavior                 |
| `--llm-kwargs`        | `{}`                      | Additional kwargs for vLLM's LLM() constructor. Supplied values override task defaults. |
| `--sampling-kwargs`   | `{}`                      | Additional kwargs for vLLM's SamplingParams() constructor. Supplied values override task defaults. |
| `--chat-kwargs`       | `{}`                      | Additional kwargs for vLLM's LLM.chat(). Supplied values override task defaults.        |


## Supported Input Formats

Text files (.txt, .text, .md, .log, .rtf) and images (.jpg, .jpeg, .png, .tiff, .tif, .bmp)

## Output Format

Plain text (.txt).

## Models

Any HuggingFace model that is compatible with vLLM's `LLM.chat()`.

## Examples

### Analyze text

=== "Config"

    ```yaml title="config.yaml"
    tasks:
        - name: chat
            kind: slurm
            module: tigerflow_ml.text.chat.slurm
            input_ext: .txt
            output_ext: .txt
            max_workers: 1
            worker_resources:
                cpus: 1
                gpus: 1
                memory: 10G
                time: 00:10:00
                sbatch_options:
                    - "--constraint=gpu80"
            setup_commands:
                - source ~/github/tigerflow-ml/.venv/bin/activate
                - export VLLM_USE_FLASHINFER_SAMPLER=0
            params:
                model: Qwen/Qwen2.5-VL-7B-Instruct
                cache-dir: ~/github/tigerflow-ml/.hf/hub/
                prompt: 'Analyze this poem'
    ```

=== "Input (.txt)"

    ```text title="Raven.txt"
    "The Raven" by Edgar Allan Poe
    ```

=== "Output (.txt)"

    ```text title="Raven.txt"
    This poem, "The Raven," by Edgar Allan Poe, is a masterful example of gothic literature, characterized by its dark, mysterious, and often melancholic themes. Here's a detailed analysis of the poem:

    ### Structure and Form
    - **Narrative Structure**: The poem is written in a first-person narrative, with the speaker recounting a series of events that unfold over a single night. The narrative is structured in a series of stanzas, each containing a distinct scene or moment of the speaker's experience.
    - **Repetition and Symbolism**: The word "Nevermore" is repeated multiple times, serving as a key symbol that ties the poem together. It represents the speaker's despair, the bird's mysterious nature, and the overall theme of futility and loss.

    ### Themes
    1. **Loss and Grief**:
    - The poem revolves around the speaker's grief over the loss of his beloved, Lenore. The speaker's longing for her is palpable, and the repetition of "Nevermore" underscores the permanence of her absence.
    - The speaker's attempts to find solace in books and memories of Lenore are futile, highlighting the depth of his sorrow.

    2. **Mystery and the Supernatural**:
    - The mysterious tapping at the door, the appearance of the raven, and the bird's cryptic response all contribute to a sense of the supernatural and the unknown.
    - The raven itself is a symbol of death and the macabre, adding to the overall gothic atmosphere.

    3. **Isolation and Despair**:
    - The speaker is isolated in his chamber, cut off from the outside world. The raven's presence and the speaker's inability to communicate with it amplify his sense of isolation and despair.
    - The poem explores the speaker's struggle to find meaning in his life, given the loss of his loved one.

    4. **The Power of Memory and Imagination**:
    - The speaker's mind wanders, reflecting on memories of Lenore and the raven. These mental wanderings are a form of escape, but they also highlight the speaker's inability to move past his grief.

    ### Symbolism
    - **The Raven**: The raven is a powerful symbol of death, mystery, and the supernatural. Its presence in the poem is both literal and metaphorical, representing the speaker's inner turmoil and the external forces that shape his life.
    - **The Door and the Window**: The door and the window are significant as
    ```


!!! note

    Notice that the output is cut off. This happens when `--max-tokens` is reached. Check the logs for any warnings of truncation if using a small max token value.


### Caption images

=== "Config"

    ```yaml title="config.yaml"
    tasks:
        - name: chat
            kind: slurm
            module: tigerflow_ml.text.chat.slurm
            input_ext: .jpeg
            output_ext: .txt
            max_workers: 1
            worker_resources:
                cpus: 1
                gpus: 1
                memory: 10G
                time: 00:10:00
                sbatch_options:
                    - "--constraint=gpu80"
            setup_commands:
                - source ~/github/tigerflow-ml/.venv/bin/activate
                - export VLLM_USE_FLASHINFER_SAMPLER=0
            params:
                model: Qwen/Qwen2.5-VL-7B-Instruct
                cache-dir: ~/github/tigerflow-ml/.hf/hub/
                prompt: 'Caption this image'
                max-image-pixels: 500
                chat-kwargs: {"use_tqdm":True}
    ```

=== "Input (.jpeg)"

    A stock image of a hummingbird and flower.

=== "Output (.txt)"

    ```text title="hummingbird.txt"
    "Nature's Symphony: A Hummingbird's Dance with a Lily"
    ```

### Structured outputs

If you're using a model which supports [structured output](https://docs.vllm.ai/en/latest/features/structured_outputs/#offline-inference), you can provide one using `--response-schema`.

=== "Choice"

    ```yaml title="config.yaml"
    tasks:
        - name: chat
            kind: slurm
            module: tigerflow_ml.text.chat.slurm
            input_ext: .txt
            output_ext: .txt
            max_workers: 1
            worker_resources:
                cpus: 1
                gpus: 1
                memory: 10G
                time: 00:10:00
                sbatch_options:
                    - "--constraint=gpu80"
            setup_commands:
                - source ~/github/tigerflow-ml/.venv/bin/activate
                - export VLLM_USE_FLASHINFER_SAMPLER=0
            params:
                model: Qwen/Qwen2.5-VL-7B-Instruct
                cache-dir: ~/github/tigerflow-ml/.hf/hub/
                prompt: 'What is the sentiment of this text?'
                response_schema: choice=["Positive", "Negative"]
    ```

=== "JSON"

    ```yaml title="config.yaml"
    tasks:
        - name: chat
            kind: slurm
            module: tigerflow_ml.text.chat.slurm
            input_ext: .jpeg
            output_ext: .txt
            max_workers: 1
            worker_resources:
                cpus: 1
                gpus: 2
                memory: 8G
                time: 01:00:00
                sbatch_options:
                    - "--constraint=gpu80"
            setup_commands:
                - source ~/github/tigerflow-ml/.venv/bin/activate
                - export VLLM_USE_FLASHINFER_SAMPLER=0
            params:
                model: Qwen/Qwen2.5-VL-32B-Instruct
                cache-dir: ~/github/tigerflow-ml/.hf/hub/
                prompt: What is in this image?
                response_schema: 'json={"type":"object","properties":{"objects":{"type":"array","items":{"type":"string"}},"scene":{"type":"string"},"dominant_colors":{"type":"array","items":{"type":"string"}}},"required":["objects","scene"]}'
                max-model-len: 4096
    ```
