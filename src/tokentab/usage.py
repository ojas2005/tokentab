"""Token usage containers and extraction from provider response objects."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Sequence

__all__ = ["TokenUsage", "extract_usage"]


def _as_int(value: Any) -> int:
    """Coerce a provider-supplied token count to a non-negative int."""
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return 0
    return 0


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    """Token counts for a single request.

    ``input_tokens`` counts only tokens billed at the full input rate. Cached
    tokens are tracked separately because providers bill them differently:
    ``cache_read_tokens`` are charged at a discount, and ``cache_write_tokens``
    (Anthropic prompt caching) are charged at a premium.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field.name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{field.name} must be non-negative, got {value}")

    @property
    def total_tokens(self) -> int:
        """Every billed token, cache traffic included. Reasoning tokens are
        excluded because providers already count them inside ``output_tokens``."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


def _get(obj: Any, *names: str) -> Any:
    """Read the first present key/attribute from a mapping or an SDK object."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return None


def _find_usage_payload(response: Any) -> Any:
    """Locate the usage object on a provider response, or return it unchanged."""
    if response is None:
        return None
    direct = _get(response, "usage", "usage_metadata", "token_usage")
    if direct is not None:
        return direct
    # OpenAI-style wrappers (LangChain LLMResult, raw dicts) nest one level deeper.
    for key in ("llm_output", "response_metadata", "additional_kwargs", "raw", "meta"):
        nested = _get(response, key)
        if nested is not None:
            found = _get(nested, "usage", "usage_metadata", "token_usage")
            if found is not None:
                return found
    return response


def extract_usage(response: Any) -> Optional[TokenUsage]:
    """Read a :class:`TokenUsage` out of a provider response or usage object.

    Understands the Anthropic Messages API, the OpenAI chat/responses APIs, the
    Google Gemini API, LangChain's ``usage_metadata``, and plain dictionaries in
    any of those shapes. Returns ``None`` when nothing usage-shaped is found, so
    callers can fall back to their own estimate instead of silently billing zero.
    """
    payload = _find_usage_payload(response)
    if payload is None:
        return None

    input_tokens = _as_int(
        _get(payload, "input_tokens", "prompt_tokens", "prompt_token_count", "promptTokenCount")
    )
    output_tokens = _as_int(
        _get(
            payload,
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
        )
    )
    cache_write = _as_int(
        _get(payload, "cache_creation_input_tokens", "cache_creation_tokens", "cache_write_tokens")
    )
    cache_read = _as_int(
        _get(
            payload,
            "cache_read_input_tokens",
            "cached_content_token_count",
            "cachedContentTokenCount",
            "cache_read_tokens",
        )
    )

    # OpenAI reports cache hits and reasoning tokens in *_tokens_details sub-objects.
    prompt_details = _get(payload, "prompt_tokens_details", "input_token_details")
    if prompt_details is not None:
        cache_read = cache_read or _as_int(_get(prompt_details, "cached_tokens", "cache_read"))
        cache_write = cache_write or _as_int(_get(prompt_details, "cache_creation"))
    completion_details = _get(payload, "completion_tokens_details", "output_token_details")
    reasoning = _as_int(_get(completion_details, "reasoning_tokens", "reasoning"))

    if not any((input_tokens, output_tokens, cache_read, cache_write, reasoning)):
        return None

    # OpenAI's prompt_tokens is inclusive of cached tokens; Anthropic's is not.
    # Subtracting when the totals overlap keeps cached tokens from being billed twice.
    if input_tokens >= cache_read + cache_write and (cache_read or cache_write):
        total = _as_int(_get(payload, "total_tokens", "total_token_count", "totalTokenCount"))
        anthropic_style = total == 0 or total >= input_tokens + cache_read + cache_write
        if not anthropic_style:
            input_tokens -= cache_read + cache_write

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
    )


def sum_usage(items: Sequence[TokenUsage]) -> TokenUsage:
    """Add up a sequence of usage records."""
    total = TokenUsage()
    for item in items:
        total = total + item
    return total
