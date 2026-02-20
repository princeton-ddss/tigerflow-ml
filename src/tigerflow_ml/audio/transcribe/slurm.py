from tigerflow.tasks import SlurmTask

from tigerflow_ml.audio.transcribe._base import _TranscribeBase


class Transcribe(_TranscribeBase, SlurmTask):
    """Audio transcription task for Slurm execution."""


if __name__ == "__main__":
    Transcribe.cli()
