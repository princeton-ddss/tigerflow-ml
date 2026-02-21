#!/usr/bin/env python
"""
Developer testing harness for OCR task.

Usage:
    python examples/ocr/run.py

This will:
1. Download sample image (cached in .cache/)
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

# Sample image: Handwritten text from IAM dataset (full paragraph)
# Used in TrOCR paper - contains multiple lines of handwritten text
IMAGE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/handwriting.jpg"
IMAGE_FILE = "handwriting_sample.png"


def setup():
    CACHE_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Download and cache test image
    cached = CACHE_DIR / IMAGE_FILE
    if not cached.exists():
        print(f"Downloading {IMAGE_URL}...")
        urllib.request.urlretrieve(IMAGE_URL, cached)

    # Copy to input dir
    dest = INPUT_DIR / IMAGE_FILE
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
