from tigerflow.tasks import SlurmTask

from tigerflow_ml.multimodal.chat._base import _ChatBase


class Chat(_ChatBase, SlurmTask):
    """Chat task for Slurm execution."""


if __name__ == "__main__":
    Chat.cli()
