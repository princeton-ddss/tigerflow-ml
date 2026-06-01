from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.text.translate.slurm import Translate

__all__ = ["Translate"]


def __getattr__(name: str):
    if name == "Translate":
        from tigerflow_ml.text.translate.slurm import Translate

        return Translate
    raise AttributeError(
        f"module 'tigerflow_ml.text.translate' has no attribute {name!r}"
    )
