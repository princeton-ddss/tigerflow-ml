from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.text.embed.slurm import Embed

__all__ = ["Embed"]


def __getattr__(name: str):
    if name == "Embed":
        from tigerflow_ml.text.embed.slurm import Embed

        return Embed
    raise AttributeError(f"module 'tigerflow_ml.text.embed' has no attribute {name!r}")
