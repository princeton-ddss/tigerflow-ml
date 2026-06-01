from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tigerflow_ml.text.chat_completion.slurm import ChatCompletion

__all__ = ["ChatCompletion"]


def __getattr__(name: str):
    if name == "ChatCompletion":
        from tigerflow_ml.text.chat_completion.slurm import ChatCompletion

        return ChatCompletion
    raise AttributeError(
        f"module 'tigerflow_ml.text.chat_completion' has no attribute {name!r}"
    )
