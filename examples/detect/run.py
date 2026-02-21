#!/usr/bin/env python
"""
Developer testing harness for object detection task.

Usage:
    python examples/detect/run.py          # run both image and video
    python examples/detect/run.py image    # image only
    python examples/detect/run.py video    # video only

This will:
1. Download sample files (cached in .cache/)
2. Copy to input/ directory
3. Run tigerflow pipeline
4. Output written to output/ for manual inspection
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
CACHE_DIR = EXAMPLE_DIR / ".cache"
OUTPUT_DIR = EXAMPLE_DIR / "output"

# COCO val2017 sample: two cats on a couch with remotes
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
IMAGE_FILE = "coco_sample.jpg"

# Short Creative Commons video clip (Blender Foundation, Big Buck Bunny)
VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
VIDEO_FILE = "big_buck_bunny.mp4"


def download(url: str, filename: str) -> Path:
    """Download a file to the cache directory if not already cached."""
    cached = CACHE_DIR / filename
    if not cached.exists():
        print(f"Downloading {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(cached, "wb") as f:
            f.write(resp.read())
    return cached


def run_pipeline(config: str, input_dir: Path):
    """Run the tigerflow pipeline with the given config and input directory."""
    config_path = EXAMPLE_DIR / config
    cmd = [
        "uv",
        "run",
        "tigerflow",
        "run",
        str(config_path),
        str(input_dir),
        str(OUTPUT_DIR),
        "--idle-timeout",
        "10",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_image():
    input_dir = EXAMPLE_DIR / "input" / "image"
    input_dir.mkdir(parents=True, exist_ok=True)

    cached = download(IMAGE_URL, IMAGE_FILE)
    dest = input_dir / IMAGE_FILE
    if not dest.exists():
        shutil.copy(cached, dest)

    run_pipeline("config.yaml", input_dir)


def run_video():
    input_dir = EXAMPLE_DIR / "input" / "video"
    input_dir.mkdir(parents=True, exist_ok=True)

    cached = download(VIDEO_URL, VIDEO_FILE)
    dest = input_dir / VIDEO_FILE
    if not dest.exists():
        shutil.copy(cached, dest)

    run_pipeline("config_video.yaml", input_dir)


if __name__ == "__main__":
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("image", "both"):
        run_image()
    if mode in ("video", "both"):
        run_video()

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("Inspect manually to verify results.")
