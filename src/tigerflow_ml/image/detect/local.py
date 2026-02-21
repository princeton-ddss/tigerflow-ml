from tigerflow.tasks import LocalTask

from tigerflow_ml.image.detect._base import _DetectBase


class Detect(_DetectBase, LocalTask):
    """Object detection task for local execution."""


if __name__ == "__main__":
    Detect.cli()
