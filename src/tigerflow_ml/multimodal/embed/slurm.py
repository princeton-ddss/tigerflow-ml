from tigerflow.tasks import SlurmTask

from tigerflow_ml.multimodal.embed._base import _EmbedBase


class Embed(_EmbedBase, SlurmTask):
    """Embed task for Slurm execution."""


if __name__ == "__main__":
    Embed.cli()
