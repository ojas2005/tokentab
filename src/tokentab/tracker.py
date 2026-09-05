"""CostTracker: the context manager that ties counting, pricing and budgets together."""

from __future__ import annotations

import contextlib
import threading
import time
import warnings
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .budget import BudgetGuard, BudgetLimits, BudgetStatus, WarningHook
from .counting import MessageLike, TokenCounter, count_message_tokens
from .exceptions import PricingWarning, UnknownModelError
from .pricing import CostBreakdown, PricingRegistry, default_registry, detect_provider
from .records import UsageRecord
from .reporting import Report
from .usage import TokenUsage, extract_usage

__all__ = [
    "CostTracker",
    "TrackedCall",
    "current_tracker",
    "default_tracker",
    "set_default_tracker",
]

_stack_var: ContextVar[Tuple[CostTracker, ...]] = ContextVar("tokentab_stack", default=())
_ambient: List[CostTracker] = []
_ambient_lock = threading.RLock()
_warned_models: Set[str] = set()


def current_tracker() -> Optional[CostTracker]:
    """The innermost active :class:`CostTracker`, or ``None``.

    Looks at the context-local stack first. Worker threads start with a fresh
    context, so trackers entered with ``ambient=True`` (the default) also
    register in a process-wide stack -- that is what makes ``with
    CostTracker(...)`` cover calls that an agent framework fans out to a thread
    pool.
    """
    stack = _stack_var.get()
    if stack:
        return stack[-1]
    with _ambient_lock:
        return _ambient[-1] if _ambient else None


class TrackedCall:
    """Handle yielded by :meth:`CostTracker.call` for one in-flight request."""

    def __init__(self, tracker: CostTracker, model: str, estimated_cost: float,
                 estimated_usage: TokenUsage, tag: Optional[str],
                 metadata: Optional[Dict[str, Any]]) -> None:
        self.tracker = tracker
        self.model = model
        self.estimated_cost = estimated_cost
        self.estimated_usage = estimated_usage
        self.tag = tag
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.response: Any = None
        self.usage: Optional[TokenUsage] = None
        self.record: Optional[UsageRecord] = None

    def set_response(self, response: Any) -> None:
        """Attach the provider response; its usage object is read on block exit."""
        self.response = response

    def set_usage(self, usage: TokenUsage) -> None:
        """Attach usage directly, for streamed calls that report it out of band."""
        self.usage = usage


class CostTracker:
    """Tracks spend for a block of work and enforces budgets against it.

    ``budget`` is the per-session ceiling in USD; ``per_request`` and ``per_day``
    add the other two. Trackers nest: an inner tracker reports its spend to the
    enclosing one, so an inner per-task budget and an outer per-run budget both
    apply.

    All state is guarded by a re-entrant lock, so concurrent calls from an agent
    framework's thread pool are counted exactly once.
    """

    def __init__(
        self,
        budget: Optional[float] = None,
        *,
        per_request: Optional[float] = None,
        per_day: Optional[float] = None,
        limits: Optional[BudgetLimits] = None,
        warn_only: bool = False,
        on_warning: Optional[WarningHook] = None,
        tag: Optional[str] = None,
        name: Optional[str] = None,
        registry: Optional[PricingRegistry] = None,
        counter: Optional[TokenCounter] = None,
        strict_pricing: bool = False,
        propagate: bool = True,
        ambient: bool = True,
    ) -> None:
        if limits is None:
            limits = BudgetLimits(
                per_request=per_request, per_session=budget, per_day=per_day
            )
        elif budget is not None or per_request is not None or per_day is not None:
            raise TypeError("pass either limits= or the budget/per_request/per_day shorthands")
        self.name = name
        self.tag = tag
        self.registry = registry if registry is not None else default_registry()
        self.counter = counter
        self.strict_pricing = strict_pricing
        self.propagate = propagate
        self.ambient = ambient
        self.guard = BudgetGuard(limits, warn_only=warn_only, on_warning=on_warning)
        self.parent: Optional[CostTracker] = None
        self._records: List[UsageRecord] = []
        self._lock = threading.RLock()
        self._token: Optional[Token[Tuple[CostTracker, ...]]] = None
        self._entered = 0

    # -- lifecycle --------------------------------------------------------

    def __enter__(self) -> CostTracker:
        outer = current_tracker()
        with self._lock:
            if self.parent is None and outer is not None and outer is not self:
                self.parent = outer
            self._entered += 1
        self._token = _stack_var.set((*_stack_var.get(), self))
        if self.ambient:
            with _ambient_lock:
                _ambient.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _stack_var.reset(self._token)
            self._token = None
        if self.ambient:
            with _ambient_lock:
                for index in range(len(_ambient) - 1, -1, -1):
                    if _ambient[index] is self:
                        del _ambient[index]
                        break
        with self._lock:
            self._entered = max(self._entered - 1, 0)

    # -- pricing helpers --------------------------------------------------

    def _price(self, model: str, usage: TokenUsage) -> CostBreakdown:
        pricing = self.registry.get(model)
        if pricing is None:
            if self.strict_pricing:
                raise UnknownModelError(model)
            if model not in _warned_models:
                _warned_models.add(model)
                warnings.warn(
                    f"No pricing for model {model!r}; counting it as $0. Register it with "
                    f"tokentab.register_model() or pass strict_pricing=True to make this "
                    f"an error.",
                    PricingWarning,
                    stacklevel=3,
                )
            return CostBreakdown()
        return pricing.cost(usage)

    def estimate(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """USD cost of the given token counts, without recording anything."""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        return self._price(model, usage).total

    def estimate_messages(
        self,
        messages: MessageLike,
        model: str,
        *,
        expected_output_tokens: int = 0,
        system: Optional[str] = None,
        tools: Optional[Any] = None,
    ) -> Tuple[TokenUsage, float]:
        """Count a prompt and price it, plus an allowance for the response."""
        input_tokens = count_message_tokens(
            messages, model, counter=self.counter, system=system, tools=tools
        )
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=expected_output_tokens)
        return usage, self._price(model, usage).total

    # -- enforcement ------------------------------------------------------

    def check(
        self,
        cost: float,
        *,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        estimated: bool = True,
    ) -> None:
        """Raise :class:`~tokentab.BudgetExceededError` if ``cost`` breaches a
        limit on this tracker or on any tracker enclosing it."""
        self.guard.check(cost, model=model, tag=tag or self.tag, estimated=estimated)
        if self.propagate and self.parent is not None:
            self.parent.check(cost, model=model, tag=tag or self.tag, estimated=estimated)

    def preflight(
        self,
        model: str,
        *,
        messages: Optional[MessageLike] = None,
        input_tokens: Optional[int] = None,
        expected_output_tokens: int = 0,
        system: Optional[str] = None,
        tools: Optional[Any] = None,
        tag: Optional[str] = None,
    ) -> Tuple[TokenUsage, float]:
        """Estimate a call's cost and enforce budgets *before* it is sent.

        Returns ``(estimated_usage, estimated_cost)``; raises
        :class:`~tokentab.BudgetExceededError` if the call must not proceed.
        """
        if input_tokens is None:
            if messages is None:
                usage = TokenUsage(output_tokens=expected_output_tokens)
            else:
                usage, _ = self.estimate_messages(
                    messages,
                    model,
                    expected_output_tokens=expected_output_tokens,
                    system=system,
                    tools=tools,
                )
        else:
            usage = TokenUsage(
                input_tokens=input_tokens, output_tokens=expected_output_tokens
            )
        cost = self._price(model, usage).total
        self.check(cost, model=model, tag=tag, estimated=True)
        return usage, cost

    # -- recording --------------------------------------------------------

    def record(
        self,
        model: str,
        usage: TokenUsage,
        *,
        tag: Optional[str] = None,
        estimated: bool = False,
        duration_s: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        check_budget: bool = False,
    ) -> UsageRecord:
        """Record a completed call and add its cost to the budget counters."""
        breakdown = self._price(model, usage)
        record = UsageRecord(
            model=model,
            usage=usage,
            breakdown=breakdown,
            tag=tag or self.tag,
            provider=detect_provider(model),
            estimated=estimated,
            duration_s=duration_s,
            metadata=dict(metadata or {}),
        )
        self._append(record, check_budget=check_budget)
        return record

    def _append(self, record: UsageRecord, *, check_budget: bool = False) -> None:
        """Store a record here and, when propagating, in every enclosing tracker."""
        with self._lock:
            self._records.append(record)
        if check_budget:
            self.guard.check(
                record.cost, model=record.model, tag=record.tag, estimated=record.estimated
            )
        self.guard.commit(record.cost)
        if self.propagate and self.parent is not None:
            self.parent._append(record, check_budget=check_budget)

    def record_response(
        self,
        response: Any,
        model: str,
        *,
        tag: Optional[str] = None,
        duration_s: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fallback_usage: Optional[TokenUsage] = None,
    ) -> Optional[UsageRecord]:
        """Read usage off a provider response and record the true cost.

        Falls back to ``fallback_usage`` (flagged ``estimated=True``) when the
        response carries no usage object, as with some streamed responses.
        Returns ``None`` if there is nothing to record.
        """
        usage = extract_usage(response)
        estimated = False
        if usage is None:
            if fallback_usage is None:
                return None
            usage, estimated = fallback_usage, True
        return self.record(
            model,
            usage,
            tag=tag,
            estimated=estimated,
            duration_s=duration_s,
            metadata=metadata,
        )

    @contextlib.contextmanager
    def call(
        self,
        model: str,
        *,
        messages: Optional[MessageLike] = None,
        input_tokens: Optional[int] = None,
        expected_output_tokens: int = 0,
        system: Optional[str] = None,
        tools: Optional[Any] = None,
        tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_estimate: bool = True,
    ) -> Iterator[TrackedCall]:
        """Guard one call: budget-check before, record the real cost after.

        >>> with tracker.call("claude-sonnet-4-5", messages=msgs) as call:
        ...     call.set_response(client.messages.create(...))

        Exceptions from the body propagate untouched and nothing is recorded,
        on the assumption that a failed call was not billed.
        """
        estimated_usage, estimated_cost = self.preflight(
            model,
            messages=messages,
            input_tokens=input_tokens,
            expected_output_tokens=expected_output_tokens,
            system=system,
            tools=tools,
            tag=tag,
        )
        handle = TrackedCall(self, model, estimated_cost, estimated_usage, tag, metadata)
        started = time.perf_counter()
        yield handle
        duration = time.perf_counter() - started
        usage = handle.usage or extract_usage(handle.response)
        estimated = False
        if usage is None:
            if not record_estimate:
                return
            usage, estimated = estimated_usage, True
        handle.record = self.record(
            model,
            usage,
            tag=tag,
            estimated=estimated,
            duration_s=duration,
            metadata=handle.metadata,
        )

    # -- results ----------------------------------------------------------

    @property
    def records(self) -> List[UsageRecord]:
        """Copy of the records collected so far."""
        with self._lock:
            return list(self._records)

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(record.cost for record in self._records)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(record.total_tokens for record in self._records)

    def status(self) -> BudgetStatus:
        """Current spend against the configured limits."""
        return self.guard.status()

    def report(self) -> Report:
        """Aggregated view of everything recorded."""
        return Report(self.records, name=self.name, limits=self.guard.limits)

    def reset(self) -> None:
        """Drop all records and zero the session counter."""
        with self._lock:
            self._records.clear()
        self.guard.reset(session=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __bool__(self) -> bool:
        """Always true. Without this, ``__len__`` would make a tracker that has
        not recorded anything yet falsy, and ``tracker or fallback`` would
        silently pick the fallback."""
        return True

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        return (
            f"<CostTracker{label} calls={len(self)} spend=${self.total_cost:.6f} "
            f"limits={self.guard.limits.to_dict()}>"
        )


_default: Optional[CostTracker] = None
_default_lock = threading.Lock()


def default_tracker() -> CostTracker:
    """Process-wide tracker used when no tracker is active or passed."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = CostTracker(name="default", ambient=False)
    return _default


def set_default_tracker(tracker: Optional[CostTracker]) -> None:
    """Replace (or with ``None``, reset) the process-wide default tracker."""
    global _default
    with _default_lock:
        _default = tracker


def resolve_tracker(explicit: Optional[CostTracker] = None) -> CostTracker:
    """Explicit tracker, else the innermost active one, else the default."""
    if explicit is not None:
        return explicit
    active = current_tracker()
    return active if active is not None else default_tracker()


def sum_records(records: Sequence[UsageRecord]) -> float:
    """Total cost of a sequence of records."""
    return sum(record.cost for record in records)
