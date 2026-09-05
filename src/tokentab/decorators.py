"""The ``@track_cost`` decorator."""

from __future__ import annotations

import functools
import inspect
import time
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast, overload

from .counting import MessageLike
from .exceptions import TokenTabError
from .tracker import CostTracker, resolve_tracker
from .usage import TokenUsage, extract_usage

__all__ = ["track_cost"]

F = TypeVar("F", bound=Callable[..., Any])

# Parameter names commonly holding the prompt, tried in order when the caller
# does not name one explicitly.
_PROMPT_PARAMS = ("messages", "prompt", "input", "text", "contents", "query")
_MODEL_PARAMS = ("model", "model_name", "model_id", "deployment")


def _bind(func: Callable[..., Any], args: Any, kwargs: Any) -> Dict[str, Any]:
    """Best-effort mapping of parameter name to value, defaults included."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):  # pragma: no cover - exotic signatures
        return dict(kwargs)


def _pick(values: Dict[str, Any], explicit: Optional[str], candidates: Tuple[str, ...]) -> Any:
    if explicit is not None:
        return values.get(explicit)
    for name in candidates:
        if values.get(name) is not None:
            return values[name]
    return None


class _Plan:
    """Everything the wrapper needs to know for one invocation."""

    __slots__ = ("cost", "model", "tag", "tracker", "usage")

    def __init__(self, tracker: CostTracker, model: Optional[str], tag: Optional[str]) -> None:
        self.tracker = tracker
        self.model = model
        self.tag = tag
        self.usage: Optional[TokenUsage] = None
        self.cost = 0.0


@overload
def track_cost(func: F) -> F: ...


@overload
def track_cost(
    func: None = ...,
    *,
    model: Optional[str] = ...,
    tag: Optional[str] = ...,
    tracker: Optional[CostTracker] = ...,
    model_arg: Optional[str] = ...,
    messages_arg: Optional[str] = ...,
    system_arg: Optional[str] = ...,
    expected_output_tokens: int = ...,
    estimate_input_tokens: Optional[Callable[..., int]] = ...,
    enforce: bool = ...,
    record_estimate: bool = ...,
    metadata: Optional[Dict[str, Any]] = ...,
) -> Callable[[F], F]: ...


def track_cost(
    func: Optional[F] = None,
    *,
    model: Optional[str] = None,
    tag: Optional[str] = None,
    tracker: Optional[CostTracker] = None,
    model_arg: Optional[str] = None,
    messages_arg: Optional[str] = None,
    system_arg: Optional[str] = "system",
    expected_output_tokens: int = 0,
    estimate_input_tokens: Optional[Callable[..., int]] = None,
    enforce: bool = True,
    record_estimate: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[F, Callable[[F], F]]:
    """Track the cost of an LLM call made inside ``func``.

    Works on sync and async functions, bare or with arguments::

        @track_cost(model="gpt-4o", tag="summarize")
        def summarize(messages):
            return client.chat.completions.create(model="gpt-4o", messages=messages)

    Before the call, the decorator estimates the prompt size (from the parameter
    named by ``messages_arg``, or the first of ``messages``/``prompt``/``input``/
    ``text``/``contents``/``query`` it finds, or ``estimate_input_tokens``) and
    asks the active tracker to enforce budgets, so an over-budget call raises
    :class:`~tokentab.BudgetExceededError` before any money is spent. Set
    ``enforce=False`` to observe without blocking.

    After the call, usage is read from the returned provider response and the
    true cost is recorded. If the response carries no usage object, the pre-call
    estimate is recorded instead and flagged ``estimated=True`` (disable with
    ``record_estimate=False``).

    Exceptions raised by the wrapped function always propagate unchanged, and
    nothing is recorded for a failed call. Errors in tokentab's own bookkeeping
    after a successful call are downgraded to warnings so they cannot destroy a
    result you have already paid for.

    ``model`` may be omitted when the wrapped function takes the model as an
    argument; the decorator reads it from ``model_arg`` or from the first of
    ``model``/``model_name``/``model_id``/``deployment`` present.
    """

    def decorate(target: F) -> F:
        is_async = inspect.iscoroutinefunction(target)

        def before(args: Any, kwargs: Any) -> _Plan:
            active = resolve_tracker(tracker)
            values = _bind(target, args, kwargs)
            resolved_model = model or _pick(values, model_arg, _MODEL_PARAMS)
            plan = _Plan(active, str(resolved_model) if resolved_model else None, tag)
            if plan.model is None:
                return plan
            input_tokens: Optional[int] = None
            messages: Optional[MessageLike] = None
            if estimate_input_tokens is not None:
                input_tokens = int(estimate_input_tokens(*args, **kwargs))
            else:
                candidate = _pick(values, messages_arg, _PROMPT_PARAMS)
                if isinstance(candidate, (str, dict, list, tuple)):
                    messages = cast(MessageLike, candidate)
            if input_tokens is None and messages is None:
                return plan
            raw_system = values.get(system_arg) if system_arg else None
            system = raw_system if isinstance(raw_system, str) else None
            if enforce:
                # preflight() prices the prompt and raises before any spend.
                usage, cost = active.preflight(
                    plan.model,
                    messages=messages,
                    input_tokens=input_tokens,
                    expected_output_tokens=expected_output_tokens,
                    system=system,
                    tag=tag,
                )
            elif input_tokens is not None:
                usage = TokenUsage(
                    input_tokens=input_tokens, output_tokens=expected_output_tokens
                )
                cost = active.estimate(plan.model, input_tokens, expected_output_tokens)
            else:
                usage, cost = active.estimate_messages(
                    cast(MessageLike, messages),
                    plan.model,
                    expected_output_tokens=expected_output_tokens,
                    system=system,
                )
            plan.usage, plan.cost = usage, cost
            return plan

        def after(plan: _Plan, result: Any, duration: float) -> None:
            if plan.model is None:
                return
            try:
                usage = extract_usage(result)
                estimated = False
                if usage is None:
                    if not record_estimate or plan.usage is None:
                        return
                    usage, estimated = plan.usage, True
                plan.tracker.record(
                    plan.model,
                    usage,
                    tag=plan.tag,
                    estimated=estimated,
                    duration_s=duration,
                    metadata=dict(metadata or {}),
                )
            except TokenTabError as exc:
                # The call already succeeded and was already billed by the
                # provider; losing the result over a bookkeeping problem would
                # be strictly worse than an inaccurate ledger.
                warnings.warn(
                    f"tokentab could not record usage: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )

        if is_async:

            @functools.wraps(target)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                plan = before(args, kwargs)
                started = time.perf_counter()
                result = await target(*args, **kwargs)
                after(plan, result, time.perf_counter() - started)
                return result

            return cast(F, async_wrapper)

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            plan = before(args, kwargs)
            started = time.perf_counter()
            result = target(*args, **kwargs)
            after(plan, result, time.perf_counter() - started)
            return result

        return cast(F, wrapper)

    if func is not None:
        return decorate(func)
    return decorate
