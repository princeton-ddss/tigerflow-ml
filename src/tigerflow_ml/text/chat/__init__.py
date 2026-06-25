from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.text.chat.slurm import Chat

__all__ = ["Chat"]


def __getattr__(name: str):
    if name == "Chat":
        from tigerflow_ml.text.chat.slurm import Chat

        return Chat
    raise AttributeError(f"module 'tigerflow_ml.text.chat' has no attribute {name!r}")
