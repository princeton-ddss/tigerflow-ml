from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.image.detect.slurm import Detect

__all__ = ["Detect"]


def __getattr__(name: str):
    if name == "Detect":
        from tigerflow_ml.image.detect.slurm import Detect

        return Detect
    raise AttributeError(
        f"module 'tigerflow_ml.image.detect' has no attribute {name!r}"
    )
