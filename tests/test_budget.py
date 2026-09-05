from __future__ import annotations

import datetime as dt
import threading

import pytest

from tokentab import BudgetExceededError, BudgetGuard, BudgetLimits, BudgetWarning


def test_per_request_limit():
    guard = BudgetGuard(BudgetLimits(per_request=0.10))
    guard.check(0.05)
    with pytest.raises(BudgetExceededError) as info:
        guard.check(0.11)
    assert info.value.limit_type == "per_request"
    assert info.value.limit == 0.10


def test_per_session_limit_accumulates():
    guard = BudgetGuard(BudgetLimits(per_session=1.0))
    for _ in range(9):
        guard.check(0.1)
        guard.commit(0.1)
    guard.check(0.1)  # lands exactly on the limit: allowed
    guard.commit(0.1)
    with pytest.raises(BudgetExceededError):
        guard.check(0.01)


def test_exactly_on_limit_is_allowed():
    guard = BudgetGuard(BudgetLimits(per_request=1.0))
    guard.check(1.0)
    with pytest.raises(BudgetExceededError):
        guard.check(1.0000001)


def test_per_day_limit_and_rollover():
    day = {"value": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)}
    guard = BudgetGuard(BudgetLimits(per_day=1.0), clock=lambda: day["value"])
    guard.commit(0.9)
    with pytest.raises(BudgetExceededError) as info:
        guard.check(0.2)
    assert info.value.limit_type == "per_day"

    day["value"] = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)
    guard.check(0.2)  # new day, counter reset
    assert guard.day_spend == 0.0


def test_session_survives_day_rollover():
    day = {"value": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)}
    guard = BudgetGuard(BudgetLimits(per_session=1.0, per_day=10.0), clock=lambda: day["value"])
    guard.commit(0.95)
    day["value"] = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)
    assert guard.day_spend == 0.0
    with pytest.raises(BudgetExceededError) as info:
        guard.check(0.1)
    assert info.value.limit_type == "per_session"


def test_warn_only_mode_does_not_raise():
    seen = []
    guard = BudgetGuard(
        BudgetLimits(per_session=0.01), warn_only=True, on_warning=seen.append
    )
    with pytest.warns(BudgetWarning, match="per_session budget exceeded"):
        guard.check(1.0)
    guard.commit(1.0)
    assert guard.session_spend == 1.0
    assert len(seen) == 1
    assert isinstance(seen[0], BudgetExceededError)


def test_unlimited_by_default():
    guard = BudgetGuard()
    assert guard.limits.is_unlimited
    guard.check(1_000_000.0)


def test_status_reports_remaining():
    guard = BudgetGuard(BudgetLimits(per_session=10.0, per_day=20.0))
    guard.commit(4.0)
    status = guard.status()
    assert status.session_spend == 4.0
    assert status.session_remaining == 6.0
    assert status.day_remaining == 16.0
    assert status.to_dict()["limits"]["per_session"] == 10.0


def test_adjust_and_reset():
    guard = BudgetGuard(BudgetLimits(per_session=10.0))
    guard.commit(5.0)
    guard.adjust(-2.0)
    assert guard.session_spend == 3.0
    guard.adjust(-100.0)  # never goes negative
    assert guard.session_spend == 0.0
    guard.commit(1.0)
    guard.reset()
    assert guard.session_spend == 0.0


def test_commit_is_thread_safe():
    guard = BudgetGuard()
    workers = 16
    per_worker = 500

    def spend():
        for _ in range(per_worker):
            guard.commit(0.001)

    threads = [threading.Thread(target=spend) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert guard.session_spend == pytest.approx(workers * per_worker * 0.001)


def test_reserve_prevents_concurrent_overspend():
    """check()+commit() can interleave; reserve() is atomic, so exactly one
    of many racing workers may claim the last of the budget."""
    guard = BudgetGuard(BudgetLimits(per_session=1.0))
    barrier = threading.Barrier(8)
    granted = []
    lock = threading.Lock()

    def attempt():
        barrier.wait()
        try:
            guard.reserve(0.5)
        except BudgetExceededError:
            return
        with lock:
            granted.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(granted) == 2
    assert guard.session_spend == pytest.approx(1.0)


def test_limits_validation():
    with pytest.raises(ValueError):
        BudgetLimits(per_session=-1)
    with pytest.raises(TypeError):
        BudgetLimits(per_day="five")  # type: ignore[arg-type]


def test_error_message_is_informative():
    error = BudgetExceededError("per_session", 1.0, 0.9, 1.5, model="gpt-4o", tag="rag")
    text = str(error)
    assert "per_session" in text and "gpt-4o" in text and "rag" in text and "estimated" in text
