from tigerflow.tasks import LocalTask

from tigerflow_ml.text.chat._base import _ChatBase


class Chat(_ChatBase, LocalTask):
    """Chat task for local execution."""


if __name__ == "__main__":
    Chat.cli()
