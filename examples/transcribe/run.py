#!/usr/bin/env python
"""
Developer testing harness for transcription task.

Usage:
    python examples/transcribe/run.py

This will:
1. Download sample audio (cached in .cache/)
2. Copy to input/ directory
3. Run tigerflow pipeline
4. Output written to output/ for manual inspection
"""

import shutil
import subprocess
import urllib.request
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
CACHE_DIR = EXAMPLE_DIR / ".cache"
INPUT_DIR = EXAMPLE_DIR / "input"
OUTPUT_DIR = EXAMPLE_DIR / "output"
CONFIG = EXAMPLE_DIR / "config.yaml"

# Sample audio: "Arthur the Rat" from DARE (Dictionary of American Regional English)
# Multiple speakers reading a phonetic passage, recorded 1965-1970
AUDIO_URL = "https://dare.wisc.edu/wp-content/uploads/sites/1051/2008/04/Arthur.mp3"
AUDIO_FILE = "arthur_the_rat.mp3"


def setup():
    CACHE_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Download and cache test audio
    cached = CACHE_DIR / AUDIO_FILE
    if not cached.exists():
        print(f"Downloading {AUDIO_URL}...")
        urllib.request.urlretrieve(AUDIO_URL, cached)

    # Copy to input dir
    dest = INPUT_DIR / AUDIO_FILE
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
