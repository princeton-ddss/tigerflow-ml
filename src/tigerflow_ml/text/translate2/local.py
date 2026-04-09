from tigerflow.tasks import LocalTask

from tigerflow_ml.text.translate2._base import _TranslateBase


class Translate(_TranslateBase, LocalTask):
    """Translation task for local execution."""


if __name__ == "__main__":
    Translate.cli()