from tigerflow.tasks import LocalTask

from tigerflow_ml.text.text_completion._base import _TextCompletionBase


class TextCompletion(_TextCompletionBase, LocalTask):
    """Translation task for local execution."""


if __name__ == "__main__":
    TextCompletion.cli()
