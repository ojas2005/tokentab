"""Budget limits and the thread-safe guard that enforces them."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import threading
import warnings
from typing import Any, Callable, Dict, Optional

from .exceptions import BudgetExceededError, BudgetWarning
from .records import utcnow

__all__ = ["BudgetLimits", "BudgetStatus", "BudgetGuard"]

WarningHook = Callable[[BudgetExceededError], None]


@dataclasses.dataclass(frozen=True)
class BudgetLimits:
    """Spend ceilings in USD. ``None`` means unlimited."""

    per_request: Optional[float] = None
    per_session: Optional[float] = None
    per_day: Optional[float] = None

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field.name} must be a number or None, got {value!r}")
            if value < 0:
                raise ValueError(f"{field.name} must be non-negative, got {value}")
            object.__setattr__(self, field.name, float(value))

    @property
    def is_unlimited(self) -> bool:
        return (
            self.per_request is None and self.per_session is None and self.per_day is None
        )

    def to_dict(self) -> Dict[str, Optional[float]]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BudgetStatus:
    """Point-in-time view of spend against limits."""

    session_spend: float
    day_spend: float
    limits: BudgetLimits
    day: str

    @property
    def session_remaining(self) -> Optional[float]:
        if self.limits.per_session is None:
            return None
        return max(self.limits.per_session - self.session_spend, 0.0)

    @property
    def day_remaining(self) -> Optional[float]:
        if self.limits.per_day is None:
            return None
        return max(self.limits.per_day - self.day_spend, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_spend": self.session_spend,
            "day_spend": self.day_spend,
            "day": self.day,
            "limits": self.limits.to_dict(),
            "session_remaining": self.session_remaining,
            "day_remaining": self.day_remaining,
        }


class BudgetGuard:
    """Enforces per-request, per-session and per-day spend limits.

    Every method takes the same lock, so concurrent workers in an agent
    framework cannot race past a limit: the check and the commit that follows it
    are individually atomic, and :meth:`reserve` makes them atomic together.
    """

    def __init__(
        self,
        limits: Optional[BudgetLimits] = None,
        *,
        warn_only: bool = False,
        on_warning: Optional[WarningHook] = None,
        clock: Optional[Callable[[], _dt.datetime]] = None,
    ) -> None:
        self.limits = limits if limits is not None else BudgetLimits()
        self.warn_only = warn_only
        self.on_warning = on_warning
        self._clock = clock or utcnow
        self._lock = threading.RLock()
        self._session_spend = 0.0
        self._day_spend = 0.0
        self._day = self._today()

    def _today(self) -> str:
        return self._clock().date().isoformat()

    def _roll_day_locked(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._day_spend = 0.0

    # -- inspection -------------------------------------------------------

    @property
    def session_spend(self) -> float:
        with self._lock:
            return self._session_spend

    @property
    def day_spend(self) -> float:
        with self._lock:
            self._roll_day_locked()
            return self._day_spend

    def status(self) -> BudgetStatus:
        with self._lock:
            self._roll_day_locked()
            return BudgetStatus(
                session_spend=self._session_spend,
                day_spend=self._day_spend,
                limits=self.limits,
                day=self._day,
            )

    # -- enforcement ------------------------------------------------------

    def _violation(self, cost: float) -> Optional[Dict[str, Any]]:
        """First breached limit for an added ``cost``, or None. Caller holds the lock."""
        checks = (
            ("per_request", self.limits.per_request, 0.0),
            ("per_session", self.limits.per_session, self._session_spend),
            ("per_day", self.limits.per_day, self._day_spend),
        )
        for name, limit, current in checks:
            if limit is None:
                continue
            projected = current + cost
            # Strictly greater: a call that lands exactly on the limit is allowed.
            if projected > limit:
                return {
                    "limit_type": name,
                    "limit": limit,
                    "current": current,
                    "projected": projected,
                }
        return None

    def _raise_or_warn(
        self,
        violation: Dict[str, Any],
        *,
        model: Optional[str],
        tag: Optional[str],
        estimated: bool,
    ) -> None:
        error = BudgetExceededError(
            limit_type=str(violation["limit_type"]),
            limit=float(violation["limit"]),
            current=float(violation["current"]),
            projected=float(violation["projected"]),
            model=model,
            tag=tag,
            estimated=estimated,
        )
        if not self.warn_only:
            raise error
        if self.on_warning is not None:
            self.on_warning(error)
        warnings.warn(str(error), BudgetWarning, stacklevel=3)

    def check(
        self,
        cost: float,
        *,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        estimated: bool = True,
    ) -> None:
        """Raise :class:`BudgetExceededError` if ``cost`` would breach a limit.

        In warn-only mode, emits a :class:`BudgetWarning` and returns normally.
        """
        with self._lock:
            self._roll_day_locked()
            violation = self._violation(cost)
        if violation is not None:
            self._raise_or_warn(violation, model=model, tag=tag, estimated=estimated)

    def commit(self, cost: float) -> None:
        """Record actual spend against the session and day counters."""
        with self._lock:
            self._roll_day_locked()
            self._session_spend += cost
            self._day_spend += cost

    def reserve(
        self,
        cost: float,
        *,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        estimated: bool = True,
    ) -> None:
        """Check and commit atomically.

        Use this when concurrent callers must not both pass a check against the
        same remaining headroom. Reconcile the estimate against the real cost
        afterwards with :meth:`adjust`.
        """
        with self._lock:
            self._roll_day_locked()
            violation = self._violation(cost)
            if violation is None:
                self._session_spend += cost
                self._day_spend += cost
        if violation is not None:
            self._raise_or_warn(violation, model=model, tag=tag, estimated=estimated)
            # warn-only mode: the spend still happens, so still record it.
            with self._lock:
                self._session_spend += cost
                self._day_spend += cost

    def adjust(self, delta: float) -> None:
        """Apply a signed correction, e.g. replacing an estimate with the real cost."""
        with self._lock:
            self._roll_day_locked()
            self._session_spend = max(self._session_spend + delta, 0.0)
            self._day_spend = max(self._day_spend + delta, 0.0)

    def reset(self, *, session: bool = True, day: bool = False) -> None:
        """Zero the counters."""
        with self._lock:
            if session:
                self._session_spend = 0.0
            if day:
                self._day = self._today()
                self._day_spend = 0.0
