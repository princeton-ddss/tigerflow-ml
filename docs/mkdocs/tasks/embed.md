# Embed

Embed text, images, audio, or video using HuggingFace sentence-transformers models.

## Parameters

| Parameter          | Default            | Description                                                                                                      |
|---------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--model`           |                    | HuggingFace model repo ID                                                                                          |
| `--revision`        | `main`             | Model revision (branch, tag, or commit hash)                                                                       |
| `--cache-dir`       |                    | HuggingFace cache directory for model files                                                                        |
| `--device`          | `auto`             | Device to use (`cuda`, `cpu`, or `auto`)                                                                            |
| `--allow-fetch`     | `--no-allow-fetch` | Allow downloads from HuggingFace Hub (network access required)                                                     |
| `--seed`            | `42`               | The seed to set for more reproducible behavior                                                                     |
| `--per-line`        | `--no-per-line`    | Embed each non-empty line of the input file independently, producing one vector per line instead of a single vector for the whole file (text input only) |
| `--batch-size`      | `32`               | Number of lines encoded per batch when `--per-line` is set, number of pages per batch for a multi-page PDF, or number of frames per batch for video |
| `--sample-fps`      | `1.0`              | Frames per second to sample from video. Set to `0` to process every frame (video input only) |
| `--prompt`          |                    | Raw text prepended to each input before encoding (e.g. `query: `). Mutually exclusive with `--prompt-name`         |
| `--prompt-name`     |                    | Name of a prompt predefined in the model's config (e.g. `query` or `passage` for e5/bge models). Mutually exclusive with `--prompt` |
| `--normalize`       | `--no-normalize`   | Whether to normalize returned vectors to have length 1                                                             |
| `--truncate-dim`    |                    | The dimension to truncate sentence embeddings to                                                                   |

## Supported Input Formats

- Text files (`.txt`, `.text`, `.md`, `.log`, `.rtf`)
- Image files (`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.heic`, `.heif`)
- PDF files (`.pdf`) — each page is rendered to an image and embedded
- Audio files (`.wav`, `.flac`, `.ogg`, `.aiff`, `.aif`, `.mp3`) — decoded, averaged to
  mono, and resampled to the rate the model expects
- Video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`) — sampled to
  frames at `--sample-fps` and each frame is embedded

## Output Format

NumPy binary (`.npy`).

- By default, a text file is embedded as one document, producing a 1-D array of shape `(dim,)`.
- With `--per-line`, each non-empty line of a text file is embedded independently, producing a 2-D array of shape `(n_lines, dim)`.
- A single image (or single-page PDF) produces a 1-D array of shape `(dim,)`.
- A multi-page PDF produces a 2-D array of shape `(n_pages, dim)`, one row per page.
- A single audio file produces a 1-D array of shape `(dim,)`.
- A video producing more than one sampled frame outputs a 2-D array of shape `(n_frames, dim)`, one row per frame; a video that samples down to a single frame outputs a 1-D array of shape `(dim,)`.

## Models

Any HuggingFace model compatible with the [sentence-transformers](https://sbert.net) library, including plain text encoder models works for text input. Embedding image or PDF input requires a multi-modal model (e.g. CLIP-style) that supports image encoding. Video is embedded as a sequence of sampled frames through the same image-capable models used for image input — there's no dedicated "video model" requirement, so any CLIP-style model works, at the cost of not modeling motion/temporal information across frames. Embedding audio requires an audio-capable model (e.g. `wav2vec2`, `HuBERT`, `WavLM`, a Whisper encoder, or CLAP) — the audio is automatically resampled to whatever rate that model's feature extractor expects.

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

### Embed audio

Use an audio-capable model to embed a sound file. The file is decoded, averaged to
mono, and resampled to the model's expected sampling rate before encoding.

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: embed
        kind: local
        module: tigerflow_ml.text.embed.local
        input_ext: .mp3
        output_ext: .npy
        params:
          model: openai/whisper-tiny
          allow-fetch: True
    ```

=== "Input"

    An audio recording, e.g. `clip.mp3`.

=== "Output (.npy)"

    A single vector of shape `(384,)`.

### Embed video

Video is sampled to frames at `--sample-fps` and each frame is embedded with the same
multi-modal model used for images.

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: embed
        kind: local
        module: tigerflow_ml.text.embed.local
        input_ext: .mp4
        output_ext: .npy
        params:
          model: sentence-transformers/clip-ViT-B-32
          sample-fps: 1.0
          allow-fetch: True
    ```

=== "Input"

    A video file, e.g. `clip.mp4` (roughly 30 seconds).

=== "Output (.npy)"

    An array of shape `(31, 512)` — one row per sampled frame (`--sample-fps 1.0`
    samples roughly one frame per second of video).

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
