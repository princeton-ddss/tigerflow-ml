# Transcription

Transcribe audio to text using HuggingFace Whisper models.

## Parameters

| Parameter             | Default                   | Description                                                   |
|-----------------------|---------------------------|---------------------------------------------------------------|
| `--model`             | `openai/whisper-large-v3` | HuggingFace model repo ID                                    |
| `--revision`          | `main`                    | Model revision (branch, tag, or commit hash)                  |
| `--cache-dir`         |                           | HuggingFace cache directory for model files                   |
| `--device`            | `auto`                    | Device to use (`cuda`, `cpu`, or `auto`)                      |
| `--language`          |                           | Source language code (e.g. `en`, `de`). Empty for auto-detect |
| `--output-format`     | `text`                    | Output format: `text`, `srt`, or `json`                       |
| `--batch-size`        | `16`                      | Batch size for processing audio chunks                        |
| `--chunk-length-s`    | `30.0`                    | Length of audio chunks in seconds                             |
| `--stride-length-s`   | `5.0`                     | Overlap between chunks in seconds                             |
| `--return-timestamps` | `false`                   | Return word/segment timestamps (required for SRT)             |

## Supported Input Formats

Audio files supported by the Whisper pipeline (WAV, MP3, FLAC, OGG, etc.).

## Output Format

Depends on `--output-format`: plain text (default), SRT subtitles, or JSON with timestamps. See [examples](#transcribe-audio-to-text) below.

## Models

Any HuggingFace [`automatic-speech-recognition`](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) model is supported. The [OpenAI Whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715571c6057) family is recommended.

| Model | Params | Speed | License |
|-------|--------|-------|---------|
| [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) (default) | 1.5B | 1x | Apache 2.0 |
| [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 809M | 3x | MIT |
| [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) | 769M | 4x | MIT |
| [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | 244M | 9x | MIT |
| [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) | 74M | 24x | MIT |
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | 39M | 50x | MIT |

!!! tip

    `whisper-large-v3-turbo` provides a good balance between accuracy and speed —
    nearly as accurate as `large-v3` at roughly 3x the speed.

## Examples

### Transcribe audio to text

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: transcribe
        kind: local
        module: tigerflow_ml.audio.transcribe.local
        input_ext: .mp3
        output_ext: .txt  # or .srt, .json
        params:
          # output_format: text   # (default) plain text
          # output_format: srt    # SRT subtitles (requires return_timestamps)
          # output_format: json   # raw Whisper output with timestamps
          # return_timestamps: true
    ```

=== "Input"

    An audio recording, e.g. `lecture.mp3`.

=== "Output (.txt)"

    ```text title="lecture.txt"
    Welcome to today's lecture on distributed computing. We will begin by
    reviewing the fundamentals of parallel processing and then move on to
    discuss fault tolerance in large-scale systems.
    ```

=== "Output (.srt)"

    ```srt title="lecture.srt"
    1
    00:00:00,000 --> 00:00:04,500
    Welcome to today's lecture on distributed computing.

    2
    00:00:04,500 --> 00:00:09,200
    We will begin by reviewing the fundamentals of parallel processing.

    3
    00:00:09,200 --> 00:00:14,800
    And then move on to discuss fault tolerance in large-scale systems.
    ```

=== "Output (.json)"

    ```json title="lecture.json"
    {
      "text": "Welcome to today's lecture on distributed computing...",
      "chunks": [
        {
          "text": "Welcome to today's lecture on distributed computing.",
          "timestamp": [0.0, 4.5]
        },
        {
          "text": "We will begin by reviewing the fundamentals of parallel processing.",
          "timestamp": [4.5, 9.2]
        }
      ]
    }
    ```

!!! note

    SRT and JSON output require `return_timestamps: true`.

### Transcribe with language hint

Setting `--language` skips auto-detection and can improve accuracy when the source
language is known:

```yaml title="config.yaml"
tasks:
  - name: transcribe
    kind: local
    module: tigerflow_ml.audio.transcribe.local
    input_ext: .mp3
    output_ext: .txt
    params:
      language: en
```

### Run on HPC with Slurm

For bulk transcription of large audio collections, use the Slurm variant to distribute
work across compute nodes:

```yaml title="config.yaml"
tasks:
  - name: transcribe
    kind: slurm
    module: tigerflow_ml.audio.transcribe.slurm
    input_ext: .mp3
    output_ext: .txt
    max_workers: 4
    worker_resources:
      cpus: 2
      gpus: 1
      memory: 16G
      time: 04:00:00
```
