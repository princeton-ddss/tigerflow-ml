from tigerflow.tasks import SlurmTask

from tigerflow_ml.text.chat_completion._base import _ChatCompletionBase


class ChatCompletion(_ChatCompletionBase, SlurmTask):
    """Chat completion task for Slurm execution."""


if __name__ == "__main__":
    ChatCompletion.cli()
