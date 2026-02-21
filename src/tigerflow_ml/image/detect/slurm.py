from tigerflow.tasks import SlurmTask

from tigerflow_ml.image.detect._base import _DetectBase


class Detect(_DetectBase, SlurmTask):
    """Object detection task for Slurm execution."""


if __name__ == "__main__":
    Detect.cli()
