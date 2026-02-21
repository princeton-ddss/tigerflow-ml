#!/usr/bin/env python
"""
Developer testing harness for translation task.

Usage:
    python examples/translate/run.py

This will:
1. Create sample English text (cached in .cache/)
2. Copy to input/ directory
3. Run tigerflow pipeline (translates English to German)
4. Output written to output/ for manual inspection
"""

import shutil
import subprocess
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
CACHE_DIR = EXAMPLE_DIR / ".cache"
INPUT_DIR = EXAMPLE_DIR / "input"
OUTPUT_DIR = EXAMPLE_DIR / "output"
CONFIG = EXAMPLE_DIR / "config.yaml"

TEXT_FILE = "sample_english.txt"

# Sample English text for translation
SAMPLE_TEXT = """\
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.

Machine translation has improved dramatically in recent years. Neural networks can now translate text with remarkable accuracy, capturing nuances that were previously impossible for computers to understand.

Today we will test the translation pipeline with this sample text. The output should be readable German text that preserves the meaning of the original English.
"""


def setup():
    CACHE_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create and cache test text
    cached = CACHE_DIR / TEXT_FILE
    if not cached.exists():
        print(f"Creating sample text: {cached}")
        with open(cached, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TEXT)

    # Copy to input dir
    dest = INPUT_DIR / TEXT_FILE
    if not dest.exists():
        shutil.copy(cached, dest)


def run():
    cmd = [
        "uv",
        "run",
        "tigerflow",
        "run",
        str(CONFIG),
        str(INPUT_DIR),
        str(OUTPUT_DIR),
        "--idle-timeout",
        "10",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    setup()
    run()
    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("Inspect manually to verify results.")
