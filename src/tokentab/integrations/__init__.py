"""Framework integrations. Each submodule imports its framework lazily."""

from __future__ import annotations

__all__ = ["TokenTabCallbackHandler"]


def __getattr__(name: str) -> object:
    if name == "TokenTabCallbackHandler":
        from .langchain import TokenTabCallbackHandler

        return TokenTabCallbackHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
