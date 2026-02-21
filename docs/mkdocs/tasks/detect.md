# Object Detection

Detect and locate objects in images and videos using HuggingFace detection models.

Supports both fixed-class models (e.g. RT-DETR) and open-vocabulary models (e.g. Grounding DINO). The pipeline type is resolved automatically from the model.

## Parameters

| Parameter      | Default              | Description                                                                 |
|----------------|----------------------|-----------------------------------------------------------------------------|
| `--model`      | `PekingU/rtdetr_r50vd` | HuggingFace model repo ID                                                |
| `--revision`   | `main`               | Model revision (branch, tag, or commit hash)                                |
| `--cache-dir`  |                      | HuggingFace cache directory for model files                                 |
| `--device`     | `auto`               | Device to use (`cuda`, `cpu`, or `auto`)                                    |
| `--labels`     |                      | Comma-separated labels for zero-shot detection (e.g. `cat,dog,person`)      |
| `--threshold`  | `0.3`                | Minimum confidence score for detections                                     |
| `--batch-size` | `4`                  | Number of video frames to process in parallel on GPU                        |
| `--sample-fps` | `1.0`                | Frames per second to sample from video (0 = every frame)                    |

## Supported Input Formats

- Image files (JPEG, PNG, TIFF, etc.)
- Video files (MP4, AVI, MOV, MKV, WebM, FLV, WMV)

## Output Format

JSON. For images, a flat array of detections:

```json
[
  {"label": "cat", "score": 0.96, "box": {"xmin": 343, "ymin": 24, "xmax": 640, "ymax": 371}},
  {"label": "remote", "score": 0.95, "box": {"xmin": 40, "ymin": 73, "xmax": 175, "ymax": 118}}
]
```

For videos, a frame-indexed array:

```json
[
  {
    "frame": 0,
    "timestamp": 0.0,
    "detections": [
      {"label": "person", "score": 0.92, "box": {"xmin": 100, "ymin": 50, "xmax": 300, "ymax": 400}}
    ]
  },
  {
    "frame": 30,
    "timestamp": 1.0,
    "detections": []
  }
]
```

## Models

### Fixed-class (COCO 80 classes)

These models detect a fixed set of object categories without needing `--labels`.

| Model | Params | COCO AP | License |
|-------|--------|---------|---------|
| `PekingU/rtdetr_r50vd` (default) | 42M | 53.1 | Apache 2.0 |
| `facebook/detr-resnet-50` | 41M | 42.0 | Apache 2.0 |

### Zero-shot (open vocabulary)

These models detect any object described by text. Requires `--labels`.

| Model | Params | License |
|-------|--------|---------|
| `IDEA-Research/grounding-dino-tiny` | 172M | Apache 2.0 |
| `IDEA-Research/grounding-dino-base` | 341M | Apache 2.0 |
| `google/owlv2-base-patch16-ensemble` | 200M | Apache 2.0 |

## Examples

Detect objects in an image using the default model:

```yaml
tasks:
  - name: detect
    kind: local
    module: ./detect.py
    input_ext: .jpg
    output_ext: .json
    params:
      model: PekingU/rtdetr_r50vd
```

Detect specific objects using a zero-shot model:

```yaml
tasks:
  - name: detect
    kind: local
    module: ./detect.py
    input_ext: .jpg
    output_ext: .json
    params:
      model: IDEA-Research/grounding-dino-base
      labels: "solar panel,wind turbine,power line"
      threshold: 0.2
```

Detect objects in video at 2 frames per second:

```yaml
tasks:
  - name: detect
    kind: local
    module: ./detect.py
    input_ext: .mp4
    output_ext: .json
    params:
      model: PekingU/rtdetr_r50vd
      sample_fps: 2.0
```
