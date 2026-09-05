"""tokentab - cost estimation and budget enforcement for LLM API calls.

Three ways in, one shared ledger::

    from tokentab import CostTracker, track_cost

    with CostTracker(budget=5.00) as tracker:        # context manager
        response = client.messages.create(...)
        tracker.record_response(response, "claude-sonnet-4-5")
        print(tracker.total_cost)

    @track_cost(model="gpt-4o", tag="summarize")      # decorator
    def summarize(messages): ...

    handler = TokenTabCallbackHandler(budget=5.00)  # LangChain
    agent.invoke(payload, config={"callbacks": [handler]})

Nothing here needs an API key, and no exception from your provider SDK is ever
caught or reshaped on its way through.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from .budget import BudgetGuard, BudgetLimits, BudgetStatus
from .counting import (
    AnthropicCounter,
    HeuristicCounter,
    TiktokenCounter,
    TokenCounter,
    count_message_tokens,
    count_tokens,
    get_counter,
    set_counter,
)
from .decorators import track_cost
from .exceptions import (
    BudgetExceededError,
    BudgetWarning,
    PricingError,
    PricingWarning,
    TokenTabError,
    UnknownModelError,
)
from .pricing import (
    CostBreakdown,
    ModelPricing,
    PricingRegistry,
    default_registry,
    detect_provider,
    normalize_model,
)
from .records import UsageRecord
from .reporting import GroupSummary, Report
from .tracker import (
    CostTracker,
    TrackedCall,
    current_tracker,
    default_tracker,
    set_default_tracker,
)
from .usage import TokenUsage, extract_usage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # tracking
    "CostTracker",
    "TrackedCall",
    "track_cost",
    "current_tracker",
    "default_tracker",
    "set_default_tracker",
    # budgets
    "BudgetGuard",
    "BudgetLimits",
    "BudgetStatus",
    # pricing
    "PricingRegistry",
    "ModelPricing",
    "CostBreakdown",
    "default_registry",
    "detect_provider",
    "normalize_model",
    "estimate_cost",
    "register_model",
    "load_pricing",
    # counting
    "TokenCounter",
    "HeuristicCounter",
    "TiktokenCounter",
    "AnthropicCounter",
    "count_tokens",
    "count_message_tokens",
    "get_counter",
    "set_counter",
    # usage and reporting
    "TokenUsage",
    "UsageRecord",
    "extract_usage",
    "Report",
    "GroupSummary",
    # errors
    "TokenTabError",
    "BudgetExceededError",
    "BudgetWarning",
    "PricingError",
    "PricingWarning",
    "UnknownModelError",
    # integrations
    "TokenTabCallbackHandler",
]


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    registry: Optional[PricingRegistry] = None,
) -> float:
    """USD cost of a call with the given token counts.

    >>> estimate_cost("claude-sonnet-4-5", 1000, 500)
    0.0105
    """
    return (registry if registry is not None else default_registry()).estimate(
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def register_model(
    model: str,
    pricing: Optional[Union[ModelPricing, Mapping[str, Any]]] = None,
    *,
    input: Optional[float] = None,  # noqa: A002 - mirrors the JSON field name
    output: Optional[float] = None,
    cache_read: Optional[float] = None,
    cache_write: Optional[float] = None,
    provider: Optional[str] = None,
    registry: Optional[PricingRegistry] = None,
) -> ModelPricing:
    """Add or override a model's prices (USD per million tokens) in the default registry.

    >>> register_model("my-finetune", input=0.5, output=1.5)
    """
    return (registry if registry is not None else default_registry()).register(
        model,
        pricing,
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        provider=provider,
    )


def load_pricing(
    source: Union[str, Mapping[str, Any]],
    *,
    registry: Optional[PricingRegistry] = None,
) -> PricingRegistry:
    """Merge a pricing JSON file (by path) or mapping into the default registry."""
    target = registry if registry is not None else default_registry()
    if isinstance(source, Mapping):
        return target.load_dict(source)
    return target.load_file(source)


def __getattr__(name: str) -> Any:
    # Keep langchain-core out of the import path for users who do not have it.
    if name == "TokenTabCallbackHandler":
        from .integrations.langchain import TokenTabCallbackHandler

        return TokenTabCallbackHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
