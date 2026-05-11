from tigerflow.tasks import SlurmTask

from tigerflow_ml.text.text_completion._base import _TextCompletionBase


class TextCompletion(_TextCompletionBase, SlurmTask):
    """Translation task for Slurm execution."""


if __name__ == "__main__":
    TextCompletion.cli()
