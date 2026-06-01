from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.audio.transcribe.slurm import Transcribe

__all__ = ["Transcribe"]


def __getattr__(name: str):
    if name == "Transcribe":
        from tigerflow_ml.audio.transcribe.slurm import Transcribe

        return Transcribe
    raise AttributeError(
        f"module 'tigerflow_ml.audio.transcribe' has no attribute {name!r}"
    )
