"""LangChain callback handler.

Drop :class:`TokenTabCallbackHandler` into any chain, agent or LLM and every
model call it makes is priced, budgeted and recorded automatically.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from ..counting import MessageLike
from ..exceptions import TokenTabError
from ..records import UsageRecord
from ..reporting import Report
from ..tracker import CostTracker, resolve_tracker
from ..usage import TokenUsage, extract_usage

__all__ = ["TokenTabCallbackHandler"]

def _resolve_base() -> Tuple[Any, bool]:
    """LangChain's callback base class, or ``object`` when it is not installed.

    Falling back to ``object`` keeps this module importable without LangChain,
    so ``tokentab.TokenTabCallbackHandler`` can be referenced anywhere and
    only *constructing* it raises a helpful ImportError.
    """
    try:  # langchain-core >= 0.1, the modern location
        from langchain_core.callbacks import BaseCallbackHandler

        return BaseCallbackHandler, True
    except ImportError:
        pass
    try:  # the legacy monolithic langchain package
        from langchain.callbacks.base import (  # type: ignore[import-not-found]
            BaseCallbackHandler as LegacyBaseCallbackHandler,
        )

        return LegacyBaseCallbackHandler, True
    except ImportError:
        return object, False


if TYPE_CHECKING:
    # Type checkers need a statically known base class; at runtime the resolver
    # above picks whichever LangChain is installed, or object.
    from langchain_core.callbacks import BaseCallbackHandler as _Base

    _HAS_LANGCHAIN = True
else:
    _Base, _HAS_LANGCHAIN = _resolve_base()


def _message_text(message: Any) -> Dict[str, Any]:
    """Normalize a LangChain message object to a role/content dict."""
    content = getattr(message, "content", message)
    role = getattr(message, "type", None) or getattr(message, "role", None) or "user"
    return {"role": str(role), "content": content}


def _model_from(serialized: Any, kwargs: Dict[str, Any]) -> str:
    """Dig the model id out of the several places LangChain puts it."""
    params = kwargs.get("invocation_params") or {}
    if isinstance(params, dict):
        for key in ("model", "model_name", "model_id", "deployment_name"):
            value = params.get(key)
            if value:
                return str(value)
    metadata = kwargs.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("ls_model_name", "model", "model_name"):
            value = metadata.get(key)
            if value:
                return str(value)
    if isinstance(serialized, dict):
        for key in ("name", "id"):
            value = serialized.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value:
                return str(value[-1])
    return "unknown"


class TokenTabCallbackHandler(_Base):
    """Prices, budgets and records every LLM call a LangChain runnable makes.

    ``on_llm_start`` / ``on_chat_model_start`` estimate the prompt and enforce
    budgets, so an over-budget step raises
    :class:`~tokentab.BudgetExceededError` before the provider is called.
    ``on_llm_end`` reads the real usage off the ``LLMResult`` and records it.

    Usage::

        handler = TokenTabCallbackHandler(budget=5.00, tag="research-agent")
        agent.invoke({"input": "..."}, config={"callbacks": [handler]})
        print(handler.report().summary())

    Pass an existing ``tracker=`` to share budgets with decorated functions and
    ``with CostTracker(...)`` blocks; otherwise the handler uses the innermost
    active tracker, or creates its own from the ``budget`` arguments.

    The pre-call estimate budgets for the response as well as the prompt, using
    ``expected_output_tokens`` if you pass one and the model's configured
    ``max_tokens`` otherwise. Without either, only the prompt is priced, and a
    session budget can overshoot by roughly one response.
    """

    raise_error = True
    """Tell LangChain to propagate exceptions from this handler, which is what
    makes ``BudgetExceededError`` actually stop the chain."""

    def __init__(
        self,
        tracker: Optional[CostTracker] = None,
        *,
        budget: Optional[float] = None,
        per_request: Optional[float] = None,
        per_day: Optional[float] = None,
        warn_only: bool = False,
        tag: Optional[str] = None,
        expected_output_tokens: int = 0,
        enforce: bool = True,
        record_estimate: bool = True,
    ) -> None:
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "TokenTabCallbackHandler needs LangChain: "
                "pip install 'tokentab[langchain]'"
            )
        super().__init__()
        if tracker is None and any(
            value is not None for value in (budget, per_request, per_day)
        ):
            tracker = CostTracker(
                budget=budget,
                per_request=per_request,
                per_day=per_day,
                warn_only=warn_only,
                tag=tag,
                name="langchain",
                ambient=False,
            )
        self._tracker = tracker
        self.tag = tag
        self.expected_output_tokens = expected_output_tokens
        self.enforce = enforce
        self.record_estimate = record_estimate
        self._lock = threading.RLock()
        self._pending: Dict[UUID, Dict[str, Any]] = {}
        self._records: List[UsageRecord] = []

    # -- plumbing ---------------------------------------------------------

    @property
    def tracker(self) -> CostTracker:
        """The tracker in use: the explicit one, else the innermost active one."""
        return resolve_tracker(self._tracker)

    def _output_allowance(self, kwargs: Dict[str, Any]) -> int:
        """Tokens to budget for the response.

        Output costs several times more than input, so estimating from the
        prompt alone lets a session budget overshoot by a whole response. When
        the caller configured ``max_tokens``, that is the true worst case, so
        use it unless an explicit allowance was passed to the handler.
        """
        if self.expected_output_tokens:
            return self.expected_output_tokens
        params = kwargs.get("invocation_params") or {}
        if isinstance(params, dict):
            for key in ("max_tokens", "max_tokens_to_sample", "max_output_tokens"):
                value = params.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return 0

    def _start(
        self,
        run_id: UUID,
        model: str,
        prompt: MessageLike,
        tag: Optional[str],
        expected_output_tokens: int = 0,
    ) -> None:
        tracker = self.tracker
        usage = TokenUsage()
        if self.enforce:
            # Raises BudgetExceededError here, before LangChain calls the model.
            usage, _ = tracker.preflight(
                model,
                messages=prompt,
                expected_output_tokens=expected_output_tokens,
                tag=tag or self.tag,
            )
        else:
            usage, _ = tracker.estimate_messages(
                prompt, model, expected_output_tokens=expected_output_tokens
            )
        with self._lock:
            self._pending[run_id] = {
                "model": model,
                "usage": usage,
                "started": time.perf_counter(),
                "tag": tag or self.tag,
            }

    def _finish(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._pending.pop(run_id, None)

    # -- LangChain callbacks ----------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id,
            _model_from(serialized, kwargs),
            list(prompts),
            tags[0] if tags else None,
            self._output_allowance(kwargs),
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        flattened = [_message_text(m) for batch in messages for m in batch]
        self._start(
            run_id,
            _model_from(serialized, kwargs),
            flattened,
            tags[0] if tags else None,
            self._output_allowance(kwargs),
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        pending = self._finish(run_id)
        if pending is None:
            return
        usage = self._usage_from_result(response)
        estimated = False
        if usage is None:
            if not self.record_estimate:
                return
            usage, estimated = pending["usage"], True
        try:
            record = self.tracker.record(
                str(pending["model"]),
                usage,
                tag=pending["tag"],
                estimated=estimated,
                duration_s=time.perf_counter() - float(pending["started"]),
            )
        except TokenTabError as exc:
            warnings.warn(
                f"tokentab could not record LangChain usage: {exc}", RuntimeWarning, stacklevel=2
            )
            return
        with self._lock:
            self._records.append(record)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Drop the pending entry. A failed call is not recorded and the error
        is left entirely to LangChain to propagate."""
        self._finish(run_id)

    @staticmethod
    def _usage_from_result(response: Any) -> Optional[TokenUsage]:
        """Find usage on an ``LLMResult``, checking every place LangChain stores it."""
        direct = extract_usage(response)
        if direct is not None:
            return direct
        generations: Sequence[Any] = getattr(response, "generations", []) or []
        for batch in generations:
            for generation in batch or []:
                for source in (
                    getattr(generation, "message", None),
                    getattr(generation, "generation_info", None),
                ):
                    if source is None:
                        continue
                    found = extract_usage(source)
                    if found is not None:
                        return found
        return None

    # -- results ----------------------------------------------------------

    @property
    def records(self) -> List[UsageRecord]:
        """Records produced by this handler (the tracker holds the full ledger)."""
        with self._lock:
            return list(self._records)

    @property
    def total_cost(self) -> float:
        return sum(record.cost for record in self.records)

    def report(self) -> Report:
        """Report over this handler's own records."""
        return Report(self.records, name="langchain", limits=self.tracker.guard.limits)
