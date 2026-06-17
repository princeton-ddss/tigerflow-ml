"""Dump raw per-window Whisper chunks (pre-merge) to inspect seam behavior.

Usage:
    python scripts/debug_windows.py /path/to/ireland1.mp3 \
        --model openai/whisper-large-v3 [--overlap-s 5.0] [--device auto]

Prints each window's index, its absolute time span, and the chunks it emitted
(with absolute timestamps), so a missing/duplicated segment at a seam can be
attributed to a specific window. No merging is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tigerflow_ml.audio.transcribe.transcriber import (
    WINDOW_S,
    BatchIterator,
    load_audio,
    load_whisper,
    process_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--overlap-s", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--language", default="")
    args = parser.parse_args()

    whisper, processor, device = load_whisper(
        args.model, "main", None, False, args.device, 42
    )

    array = load_audio(Path(args.audio))
    stride = WINDOW_S - args.overlap_s
    iterator = BatchIterator(
        array, batch_size=args.batch_size, overlap_s=args.overlap_s
    )

    language = args.language or None
    window_idx = 0
    for batch in iterator:
        offsets = [(window_idx + i) * stride for i in range(len(batch))]
        windows, language = process_batch(
            batch, whisper, processor, device, language, offsets
        )
        for w_offset, window in zip(offsets, windows):
            span_end = w_offset + WINDOW_S
            print(
                f"\n=== window {window_idx} : [{w_offset:.1f}s .. {span_end:.1f}s] ==="
            )
            for c in window.chunks:
                start, end = c.timestamp
                print(f"  [{start:7.2f} .. {end:7.2f}]  {c.text!r}")
            window_idx += 1


if __name__ == "__main__":
    main()
