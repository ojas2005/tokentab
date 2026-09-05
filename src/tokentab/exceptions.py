"""Exception and warning types raised by tokentab."""

from __future__ import annotations

from typing import Optional

__all__ = [
    "TokenTabError",
    "UnknownModelError",
    "PricingError",
    "BudgetExceededError",
    "BudgetWarning",
    "PricingWarning",
]


class TokenTabError(Exception):
    """Base class for every error raised by tokentab itself.

    Errors raised by the wrapped LLM SDK are never converted to this type;
    tokentab always lets provider exceptions propagate untouched.
    """


class PricingError(TokenTabError):
    """Raised when pricing data is malformed or cannot be loaded."""


class UnknownModelError(PricingError, KeyError):
    """Raised when a model has no pricing entry and no fallback is configured."""

    def __init__(self, model: str, message: Optional[str] = None) -> None:
        self.model = model
        text = message or (
            f"No pricing entry for model {model!r}. Register it with "
            f"tokentab.register_model({model!r}, input=..., output=...) or load a "
            f"custom pricing file, or set a fallback with PricingRegistry(fallback=...)."
        )
        super().__init__(text)

    def __str__(self) -> str:  # pragma: no cover - KeyError repr is unhelpful
        return str(self.args[0])


class BudgetExceededError(TokenTabError):
    """Raised when a call would push spend past a configured budget limit.

    Raised *before* the underlying API call whenever tokentab is given enough
    information to estimate the cost up front, so no money is spent.
    """

    def __init__(
        self,
        limit_type: str,
        limit: float,
        current: float,
        projected: float,
        *,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        estimated: bool = True,
    ) -> None:
        self.limit_type = limit_type
        self.limit = limit
        self.current = current
        self.projected = projected
        self.model = model
        self.tag = tag
        self.estimated = estimated
        qualifier = "estimated" if estimated else "actual"
        detail = f" model={model!r}" if model else ""
        detail += f" tag={tag!r}" if tag else ""
        super().__init__(
            f"{limit_type} budget exceeded: {qualifier} spend ${projected:.6f} "
            f"would pass the ${limit:.6f} limit (already spent ${current:.6f}).{detail}"
        )


class BudgetWarning(UserWarning):
    """Warning emitted instead of :class:`BudgetExceededError` in warn-only mode."""


class PricingWarning(UserWarning):
    """Warning emitted when a model has no pricing entry and its cost is counted as zero.

    Raised as :class:`UnknownModelError` instead when ``strict_pricing=True``.
    """
