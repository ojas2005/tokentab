"""tokentab - cost estimation and budget enforcement for LLM API calls."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from .exceptions import (
    BudgetExceededError,
    BudgetWarning,
    PricingError,
    PricingWarning,
    TokenTabError,
    UnknownModelError,
)
from .usage import TokenUsage, extract_usage
from .pricing import (
    CostBreakdown,
    ModelPricing,
    PricingRegistry,
    default_registry,
    detect_provider,
    normalize_model,
)
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
from .records import UsageRecord
from .budget import BudgetGuard, BudgetLimits, BudgetStatus
from .reporting import GroupSummary, Report
from .tracker import (
    CostTracker,
    TrackedCall,
    current_tracker,
    default_tracker,
    set_default_tracker,
)
from .decorators import track_cost

__version__ = "0.1.0"


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
