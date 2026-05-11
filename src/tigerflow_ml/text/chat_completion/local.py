from tigerflow.tasks import LocalTask

from tigerflow_ml.text.chat_completion._base import _ChatCompletionBase


class ChatCompletion(_ChatCompletionBase, LocalTask):
    """Translation task for local execution."""


if __name__ == "__main__":
    ChatCompletion.cli()
