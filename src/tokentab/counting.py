"""Pre-call token estimation.

Counting before the call is what lets the budget guard block a request *before*
the money is spent. Exact counts still come from the provider's usage object
after the call; everything here is an estimate, and every estimate says so.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Union

from .pricing import detect_provider, normalize_model

__all__ = [
    "TokenCounter",
    "HeuristicCounter",
    "TiktokenCounter",
    "AnthropicCounter",
    "count_tokens",
    "count_message_tokens",
    "get_counter",
    "set_counter",
    "MessageLike",
]

MessageLike = Union[str, Mapping[str, Any], Sequence[Any]]

# Rough bytes-per-token for the heuristic fallback. English prose sits near 4.0
# for every major tokenizer; code and CJK run denser, so this errs on the low
# side of cost rather than the high side of surprise.
_CHARS_PER_TOKEN = 4.0
# Per-message framing (role markers, separators) that providers bill for.
_MESSAGE_OVERHEAD = 4
_REQUEST_OVERHEAD = 3


class TokenCounter(Protocol):
    """Anything that can turn text into a token count for a given model."""

    def count(self, text: str, model: str) -> int:
        """Number of tokens ``text`` occupies for ``model``."""
        ...


class HeuristicCounter:
    """Dependency-free character-ratio estimator.

    Used whenever no exact tokenizer is installed. Typically lands within
    10-20% of the true count for prose, which is accurate enough for a budget
    guard whose job is to refuse obviously-too-expensive calls.
    """

    exact = False

    def __init__(self, chars_per_token: float = _CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str, model: str = "") -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token + 0.5))


class TiktokenCounter:
    """Exact counts for OpenAI models via the optional ``tiktoken`` extra."""

    exact = True

    def __init__(self, encoding: Optional[str] = None) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "TiktokenCounter needs tiktoken: pip install 'tokentab[tiktoken]'"
            ) from exc
        self._tiktoken = tiktoken
        self._forced = encoding
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _encoding_for(self, model: str) -> Any:
        key = self._forced or model or "cl100k_base"
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        if self._forced:
            enc = self._tiktoken.get_encoding(self._forced)
        else:
            try:
                enc = self._tiktoken.encoding_for_model(model)
            except (KeyError, ValueError):
                # Unrecognized/new model ids: o200k_base is current for GPT-4o+.
                modern = normalize_model(model).startswith(
                    ("gpt-4o", "gpt-5", "o1", "o3", "o4")
                )
                enc = self._tiktoken.get_encoding("o200k_base" if modern else "cl100k_base")
        with self._lock:
            self._cache[key] = enc
        return enc

    def count(self, text: str, model: str = "gpt-4o") -> int:
        if not text:
            return 0
        return len(self._encoding_for(model).encode(text, disallowed_special=()))


class AnthropicCounter:
    """Exact counts for Claude models via the Anthropic ``count_tokens`` endpoint.

    Needs a configured ``anthropic.Anthropic`` client. The endpoint is free but
    it is a network round trip, so results are cached per (model, text) and the
    counter degrades to :class:`HeuristicCounter` if the call fails — a budget
    guard should never be the reason an application goes down.
    """

    exact = True

    def __init__(
        self, client: Optional[Any] = None, *, fallback: Optional[TokenCounter] = None
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "AnthropicCounter needs the anthropic SDK: pip install 'tokentab[anthropic]'"
                ) from exc
            client = anthropic.Anthropic()
        self._client = client
        self._fallback: TokenCounter = fallback or HeuristicCounter()
        self._cache: Dict[str, int] = {}
        self._lock = threading.Lock()

    def count(self, text: str, model: str = "claude-sonnet-4-5") -> int:
        if not text:
            return 0
        key = f"{model}\x00{text}"
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            result = self._client.messages.count_tokens(
                model=model, messages=[{"role": "user", "content": text}]
            )
            value = int(getattr(result, "input_tokens", 0) or 0)
        except Exception:
            # Network/auth problems must not break the caller's request path.
            return self._fallback.count(text, model)
        with self._lock:
            self._cache[key] = value
        return value


_counters: Dict[str, TokenCounter] = {}
_counter_lock = threading.RLock()
_heuristic = HeuristicCounter()


def set_counter(provider_or_model: str, counter: Optional[TokenCounter]) -> None:
    """Install (or with ``None``, remove) a counter for a provider or model id.

    ``set_counter("anthropic", AnthropicCounter(client))`` wires the exact
    endpoint in for every Claude model.
    """
    key = provider_or_model.strip().lower()
    with _counter_lock:
        if counter is None:
            _counters.pop(key, None)
        else:
            _counters[key] = counter


def get_counter(model: str) -> TokenCounter:
    """Best available counter for ``model``: explicit override, then tiktoken, then heuristic."""
    name = model.strip().lower()
    provider = detect_provider(model)
    with _counter_lock:
        for key in (name, normalize_model(name), provider):
            found = _counters.get(key)
            if found is not None:
                return found
        if provider == "openai":
            try:
                counter: TokenCounter = TiktokenCounter()
            except ImportError:
                counter = _heuristic
            _counters["openai"] = counter
            return counter
    return _heuristic


def count_tokens(text: str, model: str = "", counter: Optional[TokenCounter] = None) -> int:
    """Estimate the token count of a string."""
    if not text:
        return 0
    return (counter or get_counter(model)).count(text, model)


def _content_to_text(content: Any) -> str:
    """Flatten a message content field (string, or list of typed blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "content", "input", "source"):
            if key in content:
                return _content_to_text(content[key])
        return ""
    if isinstance(content, (list, tuple)):
        return "\n".join(part for part in (_content_to_text(item) for item in content) if part)
    return str(content)


def _iter_messages(messages: MessageLike) -> Iterable[Any]:
    if isinstance(messages, (str, Mapping)):
        return [messages]
    return list(messages)


def count_message_tokens(
    messages: MessageLike,
    model: str = "",
    *,
    counter: Optional[TokenCounter] = None,
    system: Optional[str] = None,
    tools: Optional[Any] = None,
) -> int:
    """Estimate input tokens for a chat request.

    Accepts a bare string, one message dict, or a list of messages in the
    OpenAI/Anthropic ``{"role": ..., "content": ...}`` shape, including
    content-block lists. Adds the per-message framing overhead providers bill
    for, so the estimate is comparable to the ``input_tokens`` you get back.
    """
    active = counter or get_counter(model)
    items: List[Any] = list(_iter_messages(messages))
    total = 0
    for item in items:
        if isinstance(item, str):
            total += active.count(item, model) + _MESSAGE_OVERHEAD
            continue
        if isinstance(item, Mapping):
            text = _content_to_text(item.get("content", item))
            role = str(item.get("role", ""))
            name = str(item.get("name", ""))
            total += active.count(text, model) + _MESSAGE_OVERHEAD
            total += active.count(role + name, model) if (role or name) else 0
            continue
        text = _content_to_text(getattr(item, "content", item))
        total += active.count(text, model) + _MESSAGE_OVERHEAD
    if system:
        total += active.count(system, model) + _MESSAGE_OVERHEAD
    if tools is not None:
        # Tool schemas are serialized into the prompt; their JSON is a fair proxy.
        import json

        try:
            rendered = json.dumps(tools, default=str)
        except (TypeError, ValueError):  # pragma: no cover - exotic tool objects
            rendered = str(tools)
        total += active.count(rendered, model)
    return total + _REQUEST_OVERHEAD if items or system else 0
