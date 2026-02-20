from tigerflow.tasks import LocalTask

from tigerflow_ml.audio.transcribe._base import _TranscribeBase


class Transcribe(_TranscribeBase, LocalTask):
    """Audio transcription task for local execution."""


if __name__ == "__main__":
    Transcribe.cli()
