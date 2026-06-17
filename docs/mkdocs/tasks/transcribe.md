# Transcription

Transcribe audio to text using HuggingFace Whisper models.

## Parameters

| Parameter         | Default            | Description                                                                                   |
|-------------------|--------------------|-----------------------------------------------------------------------------------------------|
| `--model`         |                    | HuggingFace model repo ID (a Whisper checkpoint)                                              |
| `--revision`      | `main`             | Model revision (branch, tag, or commit hash)                                                  |
| `--cache-dir`     |                    | HuggingFace cache directory for model files                                                   |
| `--device`        | `auto`             | Device to use (`cuda`, `cpu`, or `auto`)                                                      |
| `--language`      |                    | Source language code (e.g. `en`, `de`). Empty lets the model detect it per file               |
| `--output-format` | `text`             | Output format: `text`, `srt`, `json` (all merged), or `raw` (un-merged, overlap-annotated)    |
| `--windowing`     | `batched`          | Decode strategy: `batched` (fast, overlapping windows) or `native` (seam-free, slower)        |
| `--batch-size`    | `16`               | Number of 30s windows decoded per GPU batch (batched mode)                                     |
| `--overlap-s`     | `5.0`              | Overlap (seconds) between consecutive 30s windows (batched mode)                              |
| `--allow-fetch`   | `--no-allow-fetch` | Allow downloads from HuggingFace Hub (network access required)                                 |
| `--seed`          | `42`               | Random seed for reproducibility                                                               |

## How it works

Each audio file is loaded and resampled to 16kHz mono, then transcribed with
`WhisperForConditionalGeneration`. Two decode strategies are available via
`--windowing`:

- **`batched`** (default) — the audio is cut into overlapping 30s windows
  (`--overlap-s`) that decode in parallel on the GPU, then stitched into one
  transcript. Fast. The overlap lets a word straddling a 30s boundary be
  captured by a window; stitching is **loss-averse** — it never drops a span,
  at the cost of occasionally repeating a few words at a seam.
- **`native`** — the whole file is handed to Whisper's sequential long-form
  algorithm, which advances each 30s context to the model's own last emitted
  timestamp, so boundaries fall between segments rather than mid-word. No
  seams, no merge, cleanest transcript — but it decodes sequentially and so is
  much slower than `batched`.

If you need an exact, duplicate-free transcript from the fast path, use
`--output-format raw` and reconcile the overlaps downstream (see below).

## Supported Input Formats

Common audio (and video) formats — WAV, MP3, FLAC, OGG, M4A, etc. (decoded via
`soundfile`/libsndfile).

## Output Format

Depends on `--output-format`:

- **`text`** — plain transcript text.
- **`srt`** — SubRip subtitles.
- **`json`** — merged transcript matching the
  [speech-recognition-inference](https://github.com/princeton-ddss/speech-recognition-inference)
  service schema: `{language, text, chunks: [{text, timestamp}]}`.
- **`raw`** — every window's segments, un-merged, each tagged with its `window`
  index and an `overlap` flag marking segments that fall in a region shared
  with the next window. Because overlapping windows are decoded independently
  they agree semantically but not token-for-token, so `raw` defers
  reconciliation to you (or an LLM). Drop `overlap` segments for a quick
  transcript, or reconcile both copies for an exact one.

## Models

Whisper checkpoints are supported. The
[OpenAI Whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715571c6057)
family is recommended.

| Model | Params | Speed | License |
|-------|--------|-------|---------|
| [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | 1.5B | 1x | Apache 2.0 |
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
          model: openai/whisper-large-v3
          allow_fetch: True
          # output_format: text   # (default) plain text
          # output_format: srt    # SRT subtitles
          # output_format: json   # {language, text, chunks} with timestamps
          # output_format: raw    # un-merged, overlap-annotated segments
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
      "language": "en",
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

### Cleanest transcript (slower)

When transcript quality matters more than speed, use `--windowing native`. It
avoids window seams entirely and produces no boundary duplication, at the cost
of sequential (slower) decoding.

```yaml title="config.yaml"
tasks:
  - name: transcribe
    kind: local
    module: tigerflow_ml.audio.transcribe.local
    input_ext: .mp3
    output_ext: .txt
    params:
      model: openai/whisper-large-v3
      windowing: native
      allow_fetch: True
```

### Exact reconciliation with `raw`

The `raw` format emits every window's segments with overlap annotations so you
can reconcile boundaries exactly — for example by feeding the overlapping
segments to an LLM.

```json title="recording.json (raw)"
{
  "language": "en",
  "overlap_s": 5.0,
  "segments": [
    {"text": " ...the urban accent in Sligo", "timestamp": [57.7, 59.5], "window": 1, "overlap": true},
    {"text": " which is, let's say the town...", "timestamp": [59.0, 65.7], "window": 2, "overlap": false}
  ]
}
```

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
      model: openai/whisper-large-v3
      language: en
      allow_fetch: True
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
    params:
      model: openai/whisper-large-v3
      cache_dir: ~/path/to/model/hub/
```
