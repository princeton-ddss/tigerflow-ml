from tigerflow.tasks import LocalTask

from tigerflow_ml.text.embed._base import _EmbedBase


class Embed(_EmbedBase, LocalTask):
    """Embed task for local execution."""


if __name__ == "__main__":
    Embed.cli()
