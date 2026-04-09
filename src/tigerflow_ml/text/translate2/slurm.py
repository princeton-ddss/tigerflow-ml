from tigerflow.tasks import SlurmTask

from tigerflow_ml.text.translate2._base import _TranslateBase


class Translate(_TranslateBase, SlurmTask):
    """Translation task for Slurm execution."""


if __name__ == "__main__":
    Translate.cli()