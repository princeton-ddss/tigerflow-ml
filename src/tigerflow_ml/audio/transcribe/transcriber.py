"""
Whisper transcription engine.

Ports the manual ``WhisperForConditionalGeneration`` / ``WhisperProcessor``
approach from ``princeton-ddss/speech-recognition-inference`` so that batch
transcription here produces the same ``Transcription`` contract as the live
service. Audio is cut into fixed 30s windows that decode independently (and so
batch on the GPU), and per-window timestamps are shifted into absolute time.

Unlike the service, windows overlap by ``overlap_s`` seconds, so a word
straddling a 30s boundary is captured by at least one window. (The service uses
hard, abutting cuts and can lose such words.) The engine returns the raw
per-window transcriptions; merging is deferred to the output layer so the
``raw`` format can expose every window's segments. The merged formats stitch
loss-aversely via :func:`merge_overlapping` -- never dropping a span, at the
cost of occasionally repeating a few words at a seam.
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
        segments. If generation is truncated mid-segment (e.g. it hits the
        token limit), the final segment opens with a timestamp but has no
        closing one; that tail is recovered as a segment spanning to the
        window end so it is never dropped. If no timestamps are present at all
        (e.g. a very short window), the whole string becomes one segment
        spanning the window.

        Args:
            string: Decoded text with timestamp tokens for a single window.
            language: Decode language for this window.
            offset: Seconds to add to every timestamp so the window's
                segments land in absolute file time.

        Returns:
            A ``Transcription`` for the window, with absolute timestamps.
        """
        pattern = re.compile(r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>")
        matches = list(pattern.finditer(string))

        if not matches:
            cleaned = re.sub(r"<\|.*?\|>", "", string).strip()
            return cls(
                language=language,
                text=cleaned,
                chunks=[
                    TranscriptChunk(text=cleaned, timestamp=(0.0, float(WINDOW_S)))
                ],
            ).adjust_timestamps(offset)

        chunks = [
            TranscriptChunk(
                text=m.group(2), timestamp=(float(m.group(1)), float(m.group(3)))
            )
            for m in matches
        ]

        # A trailing segment opened but never closed (truncated generation):
        # <|t|> text with no closing <|t|>. Recover it spanning to the window
        # end so its text is not dropped. Whitespace-only tails are ignored.
        tail = re.match(r"<\|(\d+\.\d+)\|>([^<]+)$", string[matches[-1].end() :])
        if tail and tail.group(2).strip():
            chunks.append(
                TranscriptChunk(
                    text=tail.group(2),
                    timestamp=(float(tail.group(1)), float(WINDOW_S)),
                )
            )

        return cls(
            language=language,
            text="".join(c.text or "" for c in chunks),
            chunks=chunks,
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


class Window(BaseModel):
    """One decoded 30s window, tagged with its position in the file.

    Args:
        index: Zero-based window index.
        offset: Absolute start time (seconds) of the window in the file.
        transcription: The window's decoded transcription (absolute timestamps).
    """

    index: int
    offset: float
    transcription: Transcription


class TranscriptionResult(BaseModel):
    """Raw, un-merged engine output: every window plus the geometry needed to
    reconcile their overlaps downstream.

    The engine intentionally does not merge. Merged output formats call
    :func:`merge_overlapping` on ``windows``; the ``raw`` format emits the
    windows as-is so a consumer (or an LLM) can reconcile overlaps itself.

    Args:
        language: Detected or forced decode language.
        windows: Per-window transcriptions, in order.
        overlap_s: Seconds of overlap between consecutive windows.
    """

    language: str | None = None
    windows: list[Window]
    overlap_s: float


def merge_overlapping(
    windows: list[Transcription],
    language: str | None,
) -> Transcription:
    """Stitch per-window transcriptions into one, loss-aversely.

    Consecutive windows overlap, so a segment near a window boundary may be
    decoded by both. A fixed seam-cut risks orphaning a segment that straddles
    the cut when the two windows disagree on where to split (ASR segment
    boundaries are not synchronized across windows). To never drop recovered
    speech, this keeps all of window ``N``'s chunks and appends a chunk from
    window ``N+1`` only when it extends past the running cursor (the end of the
    last kept chunk). A chunk fully behind the cursor is already covered and is
    skipped; a chunk that straddles the cursor is kept, which can repeat a few
    words at the seam.

    The bias is deliberate: a duplicated word at a seam is cosmetic, a dropped
    one is a real error. Consumers needing an exact, duplicate-free transcript
    should use the ``raw`` output format and reconcile overlaps themselves.

    Args:
        windows: Per-window transcriptions, in order, with absolute timestamps.
        language: Language to record on the merged transcription.

    Returns:
        A single merged ``Transcription``.
    """
    if not windows:
        return Transcription(language=language, text="", chunks=[])

    chunks: list[TranscriptChunk] = list(windows[0].chunks)
    cursor = chunks[-1].timestamp[1] if chunks else 0.0

    for window in windows[1:]:
        for c in window.chunks:
            # Keep any chunk that extends coverage past what we already have;
            # skip chunks wholly behind the cursor (already transcribed).
            if c.timestamp[1] > cursor:
                chunks.append(c)
                cursor = c.timestamp[1]

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
            # Stop once a window reaches the end: the whole file is now covered.
            # Taking another stride would start a window inside the array but
            # entirely within the previous window's reach -- a near-empty
            # trailing window that decodes as padded silence (and hallucinates).
            if self._idx + self.window_len >= len(self.array):
                self._idx = len(self.array)
                break
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

    # from_pretrained() returns the model already in inference mode and all
    # generation runs under torch.no_grad(), so switching modes is redundant.
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
) -> TranscriptionResult:
    """Transcribe an audio file into raw, un-merged windows.

    Loads and resamples the file to 16kHz mono and decodes it in overlapping
    30s windows (batched on the GPU). Merging is intentionally deferred to the
    output layer so the ``raw`` format can expose every window's segments; the
    merged formats call :func:`merge_overlapping` themselves.

    Args:
        input_file: Path to the audio (or video) file.
        whisper: The Whisper model.
        processor: The Whisper processor.
        device: Resolved device string.
        language: Forced decode language, or None to auto-detect per file.
        batch_size: Number of 30s windows decoded per GPU batch.
        overlap_s: Seconds of overlap between consecutive windows.

    Returns:
        A ``TranscriptionResult`` holding the per-window transcriptions.
    """
    array = load_audio(input_file)
    duration = len(array) / SAMPLING_RATE
    logger.info(f"Audio duration: {duration:.1f}s")

    stride = WINDOW_S - overlap_s
    iterator = BatchIterator(array, batch_size=batch_size, overlap_s=overlap_s)

    windows: list[Window] = []
    detected = language
    window_idx = 0
    for batch in iterator:
        offsets = [(window_idx + i) * stride for i in range(len(batch))]
        batch_windows, detected = process_batch(
            batch, whisper, processor, device, detected, offsets
        )
        for offset, transcription in zip(offsets, batch_windows):
            windows.append(
                Window(index=window_idx, offset=offset, transcription=transcription)
            )
            window_idx += 1
        logger.info(f"Processed {window_idx} window(s)")

    return TranscriptionResult(language=detected, windows=windows, overlap_s=overlap_s)


def transcribe_audio_native(
    input_file: Path,
    whisper: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    language: str | None,
) -> TranscriptionResult:
    """Transcribe an audio file with Whisper's native long-form algorithm.

    Instead of cutting the audio into fixed windows, the whole file is handed
    to ``generate()``, which decodes sequentially: each 30s context advances to
    the model's own last emitted timestamp, so chunk boundaries land between
    segments rather than mid-word. This avoids seams entirely (no overlap, no
    merge) and yields the cleanest transcript, but it is sequential and so much
    slower than the batched-window path.

    The result has a single window (index 0, offset 0) with no overlap, so the
    output formats treat it as already-merged.

    Args:
        input_file: Path to the audio (or video) file.
        whisper: The Whisper model.
        processor: The Whisper processor.
        device: Resolved device string.
        language: Forced decode language, or None to auto-detect.

    Returns:
        A ``TranscriptionResult`` with one window holding all segments.
    """
    import torch

    array = load_audio(input_file)
    duration = len(array) / SAMPLING_RATE
    logger.info(f"Audio duration: {duration:.1f}s (native long-form)")

    # truncation=False keeps the full audio; max_length pads up to one 30s
    # frame so short clips still satisfy the encoder's fixed 3000-frame input.
    inputs = processor(
        array,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        truncation=False,
        padding="max_length",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device, dtype=whisper.dtype)
    attention_mask = inputs.attention_mask.to(device)

    generate_kwargs: dict[str, Any] = {
        "return_timestamps": True,
        "return_segments": True,
        "task": "transcribe",
    }
    if not language:
        with torch.no_grad():
            lang_token = whisper.detect_language(input_features)[0]
        language = processor.decode([lang_token]).strip("<|>") or None
    if language:
        generate_kwargs["language"] = language

    with torch.no_grad():
        # With return_segments=True this is a dict {"sequences", "segments"},
        # not the tensor the stubs narrow to; treat as Any for indexing.
        output: Any = whisper.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    # return_segments gives one list of segments per input; we pass a single
    # file, so index 0. It can be empty for silent/undecodable audio.
    segments = output["segments"][0] if output["segments"] else []
    chunks: list[TranscriptChunk] = []
    for seg in segments:
        text = processor.decode(seg["tokens"], skip_special_tokens=True)
        start = float(seg["start"])
        end = float(seg["end"])
        chunks.append(TranscriptChunk(text=text, timestamp=(start, end)))
    logger.info(f"Decoded {len(chunks)} segment(s)")

    transcription = Transcription(
        language=language,
        text="".join(c.text or "" for c in chunks),
        chunks=chunks,
    )
    window = Window(index=0, offset=0.0, transcription=transcription)
    return TranscriptionResult(language=language, windows=[window], overlap_s=0.0)
