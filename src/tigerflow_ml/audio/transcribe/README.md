# Audio Transcription

Transcribe audio files to text, SRT subtitles, or JSON using any
[Whisper](https://huggingface.co/openai/whisper-large-v3) checkpoint on
HuggingFace.

Input is decoded with [`soundfile`](https://python-soundfile.readthedocs.io/)
(libsndfile), so the supported formats are WAV, FLAC, OGG/Vorbis, MP3, AIFF,
and the others libsndfile handles. M4A/AAC and video containers (MP4, MOV)
are **not** supported — extract the audio to a supported format first.

## How it works

1. Each file is loaded and resampled to 16kHz mono.
2. The audio is cut into overlapping 30-second windows that decode
   independently (and so batch on the GPU).
3. Each window is transcribed with `WhisperForConditionalGeneration`, producing
   segments with timestamps shifted into absolute file time.
4. Overlapping windows are stitched into a single transcription, with the
   shared region de-duplicated at the seam.
5. The transcription is written to the output directory in the requested
   format.

This mirrors the engine in
[`princeton-ddss/speech-recognition-inference`](https://github.com/princeton-ddss/speech-recognition-inference)
so batch output and the live API share one schema. The JSON format is exactly:

```json
{
  "language": "en",
  "text": "...",
  "chunks": [{ "text": "...", "timestamp": [0.0, 3.5] }]
}
```

> **Note on language detection.** Leaving `--language` empty lets Whisper detect
> the language itself during decoding. This differs from the translation task,
> which runs `langdetect` on the text *before* the model. Both are "auto", but
> the mechanism is different.

> **Note on window boundaries.** Consecutive windows overlap by `--overlap-s`
> seconds (default 5s) so a word straddling a 30s boundary is captured by at
> least one window. The merged formats (`text`/`srt`/`json`) stitch the windows
> **loss-aversely**: a span is never dropped, even if that means a few words
> repeat at a seam. If you need an exact, duplicate-free transcript, use the
> `raw` format and reconcile the overlaps yourself (or with an LLM).
>
> The upstream service uses non-overlapping windows and can lose boundary
> words; this task does not.

### Decode strategy (`--windowing`)

| Mode               | Speed                | Output                                   |
| ------------------ | -------------------- | ---------------------------------------- |
| `batched` (default) | Fast (GPU-batched)   | Lossless; some duplication at seams       |
| `native`           | Slow (sequential)    | Cleanest; no seams, no duplication        |

`batched` cuts the audio into overlapping 30s windows decoded in parallel, then
stitches them. `native` hands the whole file to Whisper's built-in long-form
algorithm, which advances each 30s context to the model's own last emitted
timestamp — so window boundaries fall between segments, never mid-word, and
there is nothing to merge. It produces the cleanest transcript but decodes
sequentially (no GPU batching within a file), so it is much slower. Use it when
you want the best single-file transcript and can wait; `--overlap-s`,
`--batch-size`, and `--raw` do not apply (native output is always a single,
seam-free window).

## Output format

The format follows the **output file extension** (`--output-ext`): `.srt` →
subtitles, `.json` → JSON, anything else → plain text. JSON matches the service
schema `{language, text, chunks}`.

### The `--raw` flag

For `.json` output, `--raw` skips merging entirely and emits every window's
segments with overlap annotations instead of the merged transcript (it is
ignored, with a warning, for non-`.json` output):

```json
{
  "language": "en",
  "overlap_s": 5.0,
  "segments": [
    {"text": " ...", "timestamp": [97.0, 102.5], "window": 3, "overlap": false},
    {"text": " ...", "timestamp": [102.5, 105.0], "window": 3, "overlap": true},
    {"text": " ...", "timestamp": [100.0, 105.4], "window": 4, "overlap": true},
    {"text": " ...", "timestamp": [105.4, 110.9], "window": 4, "overlap": false}
  ]
}
```

Segments with `"overlap": true` fall in a region shared with the next window
and have a redundant counterpart there. Because each window is decoded
independently, the two copies agree semantically but not token-for-token (ASR
segment boundaries are not synchronized across windows), which is exactly why
this is hard to merge perfectly and why `raw` defers the decision to you. Drop
the `overlap` segments for a quick transcript, or feed both copies to an LLM to
produce an exact reconciliation.

### Setup (login node)

Once you have TigerFlow ready to go, download the Whisper model you wish to use.
First, make sure you are in a directory/virtual environment with `tigerflow-ml`
installed, then:

```bash
hf auth login
HF_HOME=./.hf hf download openai/whisper-large-v3
```

Setting `HF_HOME=./.hf` defines the directory where the model's files will be
downloaded (here, a `.hf` directory under the current directory).

### Running the task

To see the available options:

```bash
python -m tigerflow_ml.audio.transcribe.slurm --help
```

A typical Slurm run:

```bash
python -m tigerflow_ml.audio.transcribe.slurm \
  --input-dir ./audio/ --input-ext .mp3 \
  --output-dir ./transcripts/ --output-ext .txt \
  --model openai/whisper-large-v3 \
  --setup-command "export HF_HOME=./.hf/" \
  --setup-command "source .venv/bin/activate" \
  --gpus 1
```

For local development, use the `local` variant:

```bash
python -m tigerflow_ml.audio.transcribe.local \
  --input-dir ./audio/ --input-ext .mp3 \
  --output-dir ./transcripts/ --output-ext .srt \
  --model openai/whisper-large-v3
```

Here the `.srt` output extension produces SRT subtitles.

## Options

| Option            | Default  | Description                                              |
| ----------------- | -------- | -------------------------------------------------------- |
| `--model`         | required | A Whisper checkpoint repo ID                             |
| `--language`      | `""`     | Source language code; empty auto-detects per file        |
| `--raw`           | `false`  | For `.json` output: emit un-merged per-window segments    |
| `--batch-size`    | `16`     | 30s windows decoded per GPU batch (batched mode)         |
| `--overlap-s`     | `5.0`    | Overlap (seconds) between windows (batched mode)          |
| `--windowing`     | `batched`| `batched` (fast) or `native` (seam-free, slow)           |

The output format is chosen from `--output-ext`: `.srt` for subtitles, `.json`
for JSON, any other extension for plain text.

Common HuggingFace options (`--revision`, `--cache-dir`, `--device`,
`--allow-fetch`, `--seed`) are also available; see `--help`.
