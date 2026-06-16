"""
Whisper transcription engine.

Ports the manual ``WhisperForConditionalGeneration`` / ``WhisperProcessor``
approach from ``princeton-ddss/speech-recognition-inference`` so that batch
transcription here produces the same ``Transcription`` contract as the live
service. Audio is cut into fixed 30s windows that decode independently (and so
batch on the GPU), and per-window timestamps are shifted into absolute time.

Unlike the service, windows overlap by ``overlap_s`` seconds and are stitched
with :func:`merge_overlapping`, which de-duplicates segments in the shared
region. The service uses hard, abutting 30s cuts, so a word straddling a 30s
boundary is split across two windows and tends to be dropped or duplicated;
the overlap + merge here recovers those boundary words while keeping windows
independent (and therefore batchable). A residual seam artifact is still
possible for a word longer than the overlap, but that is rare in speech.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from tigerflow.logconfig import logger

if TYPE_CHECKING:
    import numpy as np
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

# Whisper always operates on 16kHz mono audio and a 30s receptive field.
SAMPLING_RATE = 16000
WINDOW_S = 30


class TranscriptChunk(BaseModel):
    """A single decoded segment with an absolute ``(start, end)`` timestamp."""

    text: str | None = None
    timestamp: tuple[float, float]


class Transcription(BaseModel):
    """A full transcription: detected language, joined text, and segments.

    This mirrors the schema emitted by the speech-recognition-inference
    service so batch output and API output share one contract.
    """

    language: str | None = None
    text: str
    chunks: list[TranscriptChunk]

    @classmethod
    def from_string(
        cls, string: str, language: str | None, offset: float = 0.0
    ) -> Transcription:
        """Build a transcription from one window's decoded string.

        Whisper emits inline timestamp tokens of the form
        ``<|0.00|> some text <|2.34|>``. We parse consecutive pairs into
        segments; if none are present (e.g. a very short window), the whole
        string becomes one segment spanning the window.

        Args:
            string: Decoded text with timestamp tokens for a single window.
            language: Decode language for this window.
            offset: Seconds to add to every timestamp so the window's
                segments land in absolute file time.

        Returns:
            A ``Transcription`` for the window, with absolute timestamps.
        """
        pattern = re.compile(r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>")
        matches = pattern.findall(string)

        if not matches:
            cleaned = re.sub(r"<\|.*?\|>", "", string).strip()
            return cls(
                language=language,
                text=cleaned,
                chunks=[
                    TranscriptChunk(text=cleaned, timestamp=(0.0, float(WINDOW_S)))
                ],
            ).adjust_timestamps(offset)

        return cls(
            language=language,
            text="".join(text for _, text, _ in matches),
            chunks=[
                TranscriptChunk(text=text, timestamp=(float(start), float(end)))
                for start, text, end in matches
            ],
        ).adjust_timestamps(offset)

    def adjust_timestamps(self, offset: float = 0.0) -> Transcription:
        """Shift every chunk's timestamp by ``offset`` seconds, in place."""
        if offset:
            self.chunks = [
                TranscriptChunk(
                    text=c.text,
                    timestamp=(offset + c.timestamp[0], offset + c.timestamp[1]),
                )
                for c in self.chunks
            ]
        return self


def merge_overlapping(
    windows: list[Transcription],
    language: str | None,
    overlap_s: float,
) -> Transcription:
    """Stitch per-window transcriptions, de-duplicating the overlap region.

    Windows ``N`` and ``N+1`` share an ``overlap_s``-second region. A segment
    in that region is decoded by both windows; we keep window ``N``'s segments
    up to the seam midpoint and window ``N+1``'s segments from the midpoint on.
    Preferring whichever window the segment is more interior to favors the
    cleaner (non-truncated) copy of a boundary word.

    Args:
        windows: Per-window transcriptions, in order, with absolute timestamps.
        language: Language to record on the merged transcription.
        overlap_s: Seconds of overlap between consecutive windows.

    Returns:
        A single merged ``Transcription``.
    """
    if not windows:
        return Transcription(language=language, text="", chunks=[])

    stride = WINDOW_S - overlap_s
    chunks: list[TranscriptChunk] = list(windows[0].chunks)

    for k, window in enumerate(windows[1:]):
        # Window k spans [k*stride, k*stride + WINDOW_S]; window k+1 starts at
        # (k+1)*stride. Their overlap is [(k+1)*stride, k*stride + WINDOW_S];
        # the seam is its midpoint. Keep already-collected segments before the
        # seam and this window's segments from the seam on, so boundary words
        # are taken from whichever window decoded them more interior.
        overlap_start = (k + 1) * stride
        overlap_end = k * stride + WINDOW_S
        seam = (overlap_start + overlap_end) / 2
        chunks = [c for c in chunks if c.timestamp[0] < seam]
        chunks.extend(c for c in window.chunks if c.timestamp[0] >= seam)

    text = "".join(c.text or "" for c in chunks)
    return Transcription(language=language, text=text, chunks=chunks)


class BatchIterator:
    """Yield overlapping 30s windows of an audio array as GPU batches.

    Windows advance by ``WINDOW_S - overlap_s`` so consecutive windows share
    an ``overlap_s``-second region (see module docstring). Each ``__next__``
    returns up to ``batch_size`` windows, decoded together downstream.
    """

    def __init__(
        self,
        array: np.ndarray,
        batch_size: int = 1,
        overlap_s: float = 5.0,
    ):
        self.array = array
        self.batch_size = batch_size
        self.window_len = SAMPLING_RATE * WINDOW_S
        self.stride = int(SAMPLING_RATE * (WINDOW_S - overlap_s))
        self._idx = 0

    def __iter__(self) -> BatchIterator:
        return self

    def __next__(self) -> list[np.ndarray]:
        if self._idx >= len(self.array):
            raise StopIteration
        batch = []
        while len(batch) < self.batch_size and self._idx < len(self.array):
            batch.append(self.array[self._idx : self._idx + self.window_len])
            self._idx += self.stride
        return batch


def load_whisper(
    model: str,
    revision: str,
    cache_dir: str | None,
    allow_fetch: bool,
    device: str,
    seed: int,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, str]:
    """Load a Whisper model and processor onto the target device.

    Args:
        model: HuggingFace model repo ID (a Whisper checkpoint).
        revision: Model revision (branch, tag, or commit hash).
        cache_dir: HuggingFace cache directory, or None for the default.
        allow_fetch: Allow downloading from the Hub; otherwise local-only.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        seed: Seed for reproducible decoding.

    Returns:
        The model, processor, and the resolved device string.

    Raises:
        RuntimeError: If the model is not cached and ``allow_fetch`` is False.
    """
    import torch
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    set_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Loading Whisper model: {model}")
    logger.info(f"Device: {device}, dtype: {torch_dtype}")

    cache_dir = cache_dir or None
    local_files_only = not allow_fetch

    try:
        processor = WhisperProcessor.from_pretrained(
            model,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        whisper = WhisperForConditionalGeneration.from_pretrained(
            model,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            device_map=device,
        )
    except OSError as e:
        if local_files_only:
            raise RuntimeError(
                f"'{model}' not found in cache ({cache_dir}). "
                "Run with --allow-fetch or download manually."
            ) from e
        raise

    whisper.eval()
    logger.info("Model ready")
    return whisper, processor, device


def process_batch(
    batch: list[np.ndarray],
    whisper: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    language: str | None,
    offsets: list[float],
) -> tuple[list[Transcription], str | None]:
    """Transcribe one batch of 30s windows.

    Args:
        batch: List of audio windows (16kHz mono arrays).
        whisper: The Whisper model.
        processor: The Whisper processor.
        device: Resolved device string.
        language: Forced decode language, or None to auto-detect.
        offsets: Absolute start time (seconds) of each window in the batch,
            one per element of ``batch``.

    Returns:
        A ``(windows, language)`` pair: one ``Transcription`` per input window
        with absolute timestamps, and the (possibly auto-detected) language.
    """
    import torch

    inputs = processor(
        batch,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device, dtype=whisper.dtype)
    attention_mask = inputs.attention_mask.to(device)

    generate_kwargs: dict[str, Any] = {"return_timestamps": True, "task": "transcribe"}
    if not language:
        # Detect from the first window so all windows decode consistently.
        with torch.no_grad():
            lang_token = whisper.detect_language(input_features)[0]
        language = processor.decode([lang_token]).strip("<|>") or None
    if language:
        generate_kwargs["language"] = language

    with torch.no_grad():
        predicted_ids = whisper.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    # generate() may return a GenerateOutput wrapper or a bare tensor, and the
    # tensor is (batch, seq) for short-form or (batch, num_segments, seq) when
    # the model routes into long-form. Normalize to a list of 1-D id sequences,
    # one per input window, concatenating any per-window segments.
    if hasattr(predicted_ids, "sequences"):
        predicted_ids = predicted_ids.sequences
    sequences = _flatten_sequences(predicted_ids)

    # Decode per window. batch_decode(..., decode_with_timestamps=True) is
    # broken in transformers 5.x (it mishandles the batch dimension), so decode
    # each sequence individually.
    decoded = [
        processor.decode(seq, skip_special_tokens=True, decode_with_timestamps=True)
        for seq in sequences
    ]

    windows = [
        Transcription.from_string(string, language=language, offset=offset)
        for string, offset in zip(decoded, offsets)
    ]
    return windows, language


def _flatten_sequences(predicted_ids: Any) -> list[list[int]]:
    """Normalize generate() output to one 1-D id sequence per input window.

    ``generate()`` returns ``(batch, seq)`` for short-form decoding and
    ``(batch, num_segments, seq)`` when a window routes into long-form. In the
    3-D case the per-window segments are concatenated into a single sequence so
    each window yields exactly one decoded string.

    Args:
        predicted_ids: A 2-D or 3-D tensor (or nested list) of token ids.

    Returns:
        A list with one flat list of token ids per window.
    """
    ids = predicted_ids.tolist() if hasattr(predicted_ids, "tolist") else predicted_ids
    sequences: list[list[int]] = []
    for item in ids:
        if item and isinstance(item[0], list):
            # 3-D: concatenate this window's segments.
            sequences.append([token for segment in item for token in segment])
        else:
            sequences.append(item)
    return sequences


def load_audio(input_file: Path) -> np.ndarray:
    """Load an audio file as a 16kHz mono float32 array.

    Decodes with ``soundfile``, averages channels to mono, and resamples to
    16kHz with ``soxr`` if needed.

    Args:
        input_file: Path to the audio file.

    Returns:
        A 1-D float32 array of samples at 16kHz.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    array, sr = sf.read(str(input_file), dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != SAMPLING_RATE:
        array = soxr.resample(array, sr, SAMPLING_RATE)
    return np.ascontiguousarray(array, dtype=np.float32)


def transcribe_audio(
    input_file: Path,
    whisper: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    language: str | None,
    batch_size: int,
    overlap_s: float,
) -> Transcription:
    """Transcribe an audio file end to end.

    Loads and resamples the file to 16kHz mono, decodes it in overlapping 30s
    windows (batched on the GPU), and merges the windows into one
    ``Transcription`` with de-duplicated boundary segments.

    Args:
        input_file: Path to the audio (or video) file.
        whisper: The Whisper model.
        processor: The Whisper processor.
        device: Resolved device string.
        language: Forced decode language, or None to auto-detect per file.
        batch_size: Number of 30s windows decoded per GPU batch.
        overlap_s: Seconds of overlap between consecutive windows.

    Returns:
        The merged ``Transcription``.
    """
    array = load_audio(input_file)
    duration = len(array) / SAMPLING_RATE
    logger.info(f"Audio duration: {duration:.1f}s")

    stride = WINDOW_S - overlap_s
    iterator = BatchIterator(array, batch_size=batch_size, overlap_s=overlap_s)

    windows: list[Transcription] = []
    detected = language
    window_idx = 0
    for batch in iterator:
        offsets = [(window_idx + i) * stride for i in range(len(batch))]
        batch_windows, detected = process_batch(
            batch, whisper, processor, device, detected, offsets
        )
        windows.extend(batch_windows)
        window_idx += len(batch)
        logger.info(f"Processed {window_idx} window(s)")

    return merge_overlapping(windows, language=detected, overlap_s=overlap_s)
