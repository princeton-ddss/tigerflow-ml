# Audio Transcription

Transcribe audio (and video) files to text, SRT subtitles, or JSON using any
[Whisper](https://huggingface.co/openai/whisper-large-v3) checkpoint on
HuggingFace.

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
> seconds (default 5s) and the overlap is de-duplicated when stitching, so a
> word straddling a 30s boundary is recovered from whichever window decoded it
> more interior. The upstream service uses non-overlapping windows and can lose
> such words; this task does not.

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
  --model openai/whisper-large-v3 --output-format srt
```

## Options

| Option            | Default  | Description                                              |
| ----------------- | -------- | -------------------------------------------------------- |
| `--model`         | required | A Whisper checkpoint repo ID                             |
| `--language`      | `""`     | Source language code; empty auto-detects per file        |
| `--output-format` | `text`   | `text`, `srt`, or `json`                                 |
| `--batch-size`    | `16`     | 30s windows decoded per GPU batch                        |
| `--overlap-s`     | `5.0`    | Overlap (seconds) between windows, de-duplicated on merge |

Common HuggingFace options (`--revision`, `--cache-dir`, `--device`,
`--allow-fetch`, `--seed`) are also available; see `--help`.
