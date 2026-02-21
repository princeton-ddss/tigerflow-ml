# Object Detection

Detect and locate objects in images and videos using HuggingFace detection models.

Supports both fixed-class models (e.g. RT-DETR) and open-vocabulary models
(e.g. Grounding DINO). The pipeline type is resolved automatically from the model.

## Parameters

| Parameter      | Default                | Description                                                            |
|----------------|------------------------|------------------------------------------------------------------------|
| `--model`      | `PekingU/rtdetr_r50vd` | HuggingFace model repo ID                                             |
| `--revision`   | `main`                 | Model revision (branch, tag, or commit hash)                           |
| `--cache-dir`  |                        | HuggingFace cache directory for model files                            |
| `--device`     | `auto`                 | Device to use (`cuda`, `cpu`, or `auto`)                               |
| `--labels`     |                        | Comma-separated labels for zero-shot detection (e.g. `cat,dog,person`) |
| `--threshold`  | `0.3`                  | Minimum confidence score for detections                                |
| `--batch-size` | `4`                    | Number of video frames to process in parallel on GPU                   |
| `--sample-fps` | `1.0`                  | Frames per second to sample from video (0 = every frame)               |

## Supported Input Formats

- Image files (JPEG, PNG, TIFF, etc.)
- Video files (MP4, AVI, MOV, MKV, WebM, FLV, WMV)

## Output Format

JSON. The output structure depends on the input type.

**Images** produce a flat array of detections:

```json
[
  {"label": "cat", "score": 0.96, "box": {"xmin": 343, "ymin": 24, "xmax": 640, "ymax": 371}},
  {"label": "remote", "score": 0.95, "box": {"xmin": 40, "ymin": 73, "xmax": 175, "ymax": 118}}
]
```

**Videos** produce a frame-indexed array:

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

Any HuggingFace [`object-detection`](https://huggingface.co/models?pipeline_tag=object-detection) or [`zero-shot-object-detection`](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection) model is supported.

### Fixed-class (COCO 80 classes)

These models detect a fixed set of object categories without needing `--labels`.

| Model | Params | COCO AP | License |
|-------|--------|---------|---------|
| [`PekingU/rtdetr_r50vd`](https://huggingface.co/PekingU/rtdetr_r50vd) (default) | 42M | 53.1 | Apache 2.0 |
| [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50) | 41M | 42.0 | Apache 2.0 |

### Zero-shot (open vocabulary)

These models detect any object described by text. Requires `--labels`.

| Model | Params | License |
|-------|--------|---------|
| [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | 172M | Apache 2.0 |
| [`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base) | 341M | Apache 2.0 |
| [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble) | 200M | Apache 2.0 |

## Examples

### Detect objects in images

Uses the default RT-DETR model which recognizes 80 common object categories (COCO classes).

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: detect
        kind: local
        module: tigerflow_ml.image.detect.local
        input_ext: .jpg
        output_ext: .json
    ```

=== "Input"

    ![Input image](../assets/img/detect_input.jpg)

=== "Output"

    ![Annotated output](../assets/img/detect_output.jpg)

    ```json title="photo.json"
    [
      {"label": "sofa", "score": 0.97, "box": {"xmin": 0, "ymin": 0, "xmax": 640, "ymax": 476}},
      {"label": "cat", "score": 0.96, "box": {"xmin": 343, "ymin": 24, "xmax": 640, "ymax": 371}},
      {"label": "cat", "score": 0.96, "box": {"xmin": 13, "ymin": 54, "xmax": 318, "ymax": 472}},
      {"label": "remote", "score": 0.95, "box": {"xmin": 40, "ymin": 73, "xmax": 175, "ymax": 118}},
      {"label": "remote", "score": 0.92, "box": {"xmin": 333, "ymin": 76, "xmax": 369, "ymax": 186}}
    ]
    ```

### Detect custom objects with zero-shot

Use an open-vocabulary model to detect arbitrary objects described by text labels.

!!! note

    Zero-shot models require the `--labels` parameter. The model will search for
    objects matching the provided text descriptions.

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: detect
        kind: local
        module: tigerflow_ml.image.detect.local
        input_ext: .jpg
        output_ext: .json
        params:
          model: IDEA-Research/grounding-dino-base
          labels: "solar panel,wind turbine,power line"
          threshold: 0.2
    ```

=== "Input"

    A satellite or aerial image, e.g. `site_survey.jpg`.

=== "Output"

    ```json title="site_survey.json"
    [
      {
        "label": "solar panel",
        "score": 0.84,
        "box": {"xmin": 120, "ymin": 200, "xmax": 350, "ymax": 310}
      },
      {
        "label": "power line",
        "score": 0.71,
        "box": {"xmin": 0, "ymin": 50, "xmax": 640, "ymax": 65}
      }
    ]
    ```

### Detect objects in video

The task automatically extracts frames from video at the specified sample rate and
runs detection on each frame.

=== "Config"

    ```yaml title="config.yaml"
    tasks:
      - name: detect
        kind: local
        module: tigerflow_ml.image.detect.local
        input_ext: .mp4
        output_ext: .json
        params:
          sample_fps: 2.0
          batch_size: 8
    ```

=== "Input"

    A video file, e.g. `traffic.mp4` (30 fps, 10 seconds).

=== "Output"

    ```json title="traffic.json"
    [
      {
        "frame": 0,
        "timestamp": 0.0,
        "detections": [
          {"label": "car", "score": 0.94, "box": {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 350}},
          {"label": "person", "score": 0.91, "box": {"xmin": 400, "ymin": 150, "xmax": 450, "ymax": 320}}
        ]
      },
      {
        "frame": 15,
        "timestamp": 0.5,
        "detections": [
          {"label": "car", "score": 0.92, "box": {"xmin": 150, "ymin": 200, "xmax": 350, "ymax": 350}}
        ]
      }
    ]
    ```

    At `sample_fps: 2.0`, the task samples 2 frames per second from the video (20 frames
    total for a 10-second clip). Increase `--batch-size` to process more frames in parallel
    on the GPU.

### Run on HPC with Slurm

For bulk detection across large image or video collections, use the Slurm variant to
distribute work across compute nodes:

```yaml title="config.yaml"
tasks:
  - name: detect
    kind: slurm
    module: tigerflow_ml.image.detect.slurm
    input_ext: .mp4
    output_ext: .json
    max_workers: 4
    worker_resources:
      cpus: 2
      gpus: 1
      memory: 16G
      time: 04:00:00
    params:
      sample_fps: 1.0
      batch_size: 8
```
