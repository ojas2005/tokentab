from __future__ import annotations

import threading

import pytest

from tokentab import (
    BudgetExceededError,
    CostTracker,
    PricingWarning,
    TokenUsage,
    UnknownModelError,
    current_tracker,
)


def test_context_manager_records_and_totals(anthropic_response):
    with CostTracker(budget=5.0) as tracker:
        tracker.record_response(anthropic_response(1000, 500), "claude-sonnet-4-5")
        tracker.record_response(anthropic_response(2000, 100), "claude-sonnet-4-5")
    assert len(tracker.records) == 2
    # (1000+2000)/1M*3 + (500+100)/1M*15
    assert tracker.total_cost == pytest.approx(0.009 + 0.009)


def test_current_tracker_inside_and_outside():
    assert current_tracker() is None
    with CostTracker() as outer:
        assert current_tracker() is outer
        with CostTracker() as inner:
            assert current_tracker() is inner
        assert current_tracker() is outer
    assert current_tracker() is None


def test_preflight_blocks_before_the_call():
    tracker = CostTracker(per_request=0.001)
    with tracker:
        with pytest.raises(BudgetExceededError) as info:
            # ~1M characters of prompt is far past a $0.001 request budget.
            tracker.preflight("claude-opus-4-1", messages="x" * 1_000_000)
    assert info.value.estimated is True
    assert info.value.limit_type == "per_request"
    # Nothing was recorded, because nothing was spent.
    assert tracker.total_cost == 0.0


def test_call_context_manager_records_actual(openai_response):
    tracker = CostTracker(budget=1.0)
    with tracker:
        with tracker.call("gpt-4o", messages="summarize this", tag="demo") as call:
            assert call.estimated_cost >= 0
            call.set_response(openai_response(p=1000, c=200))
    record = tracker.records[0]
    assert record.estimated is False
    assert record.tag == "demo"
    assert record.input_tokens == 1000
    assert record.cost == pytest.approx(1000 / 1e6 * 2.5 + 200 / 1e6 * 10)
    assert record.duration_s is not None


def test_call_falls_back_to_estimate_when_no_usage():
    tracker = CostTracker()
    with tracker:
        with tracker.call("gpt-4o", messages="hello there") as call:
            call.set_response("a streamed string with no usage object")
    assert len(tracker.records) == 1
    assert tracker.records[0].estimated is True


def test_call_records_nothing_and_reraises_on_failure():
    tracker = CostTracker()
    with tracker:
        with pytest.raises(ZeroDivisionError):
            with tracker.call("gpt-4o", messages="hi"):
                raise ZeroDivisionError("provider blew up")
    assert tracker.records == []


def test_nested_trackers_propagate_spend(anthropic_response):
    with CostTracker(budget=10.0, name="run") as outer:
        with CostTracker(budget=1.0, name="task") as inner:
            inner.record_response(anthropic_response(1000, 500), "claude-sonnet-4-5")
        assert len(inner.records) == 1
        assert len(outer.records) == 1
        assert outer.total_cost == pytest.approx(inner.total_cost)


def test_inner_budget_does_not_leak_outward(anthropic_response):
    with CostTracker(budget=10.0) as outer:
        with CostTracker(budget=0.0001) as inner:
            with pytest.raises(BudgetExceededError):
                inner.check(1.0)
        outer.check(1.0)  # outer still has room


def test_outer_budget_binds_inner_calls():
    with CostTracker(budget=0.001):
        with CostTracker(budget=100.0) as inner:
            with pytest.raises(BudgetExceededError) as info:
                inner.check(1.0)
    assert info.value.limit == pytest.approx(0.001)


def test_ambient_tracker_visible_from_worker_threads(anthropic_response):
    """Agent frameworks fan calls out to threads, which get a fresh context;
    the ambient stack is what keeps those calls inside the budget."""
    seen = []
    with CostTracker(budget=5.0) as tracker:
        def work():
            seen.append(current_tracker())

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert seen == [tracker] * 4


def test_concurrent_recording_counts_every_call(anthropic_response):
    tracker = CostTracker()
    with tracker:
        def work():
            for _ in range(50):
                tracker.record_response(anthropic_response(1000, 0), "claude-sonnet-4-5")

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert len(tracker.records) == 400
    assert tracker.total_cost == pytest.approx(400 * 1000 / 1e6 * 3)


def test_unknown_model_warns_by_default():
    tracker = CostTracker()
    with pytest.warns(PricingWarning, match="No pricing for model"):
        record = tracker.record("mystery-model-x", TokenUsage(input_tokens=100))
    assert record.cost == 0.0


def test_unknown_model_raises_when_strict():
    tracker = CostTracker(strict_pricing=True)
    with pytest.raises(UnknownModelError):
        tracker.record("mystery-model-y", TokenUsage(input_tokens=100))


def test_record_response_without_usage_returns_none():
    tracker = CostTracker()
    assert tracker.record_response({"choices": []}, "gpt-4o") is None
    assert tracker.records == []


def test_record_response_uses_fallback_usage():
    tracker = CostTracker()
    record = tracker.record_response(
        "no usage", "gpt-4o", fallback_usage=TokenUsage(input_tokens=100, output_tokens=10)
    )
    assert record is not None and record.estimated is True


def test_warn_only_tracker_keeps_going():
    from tokentab import BudgetWarning

    tracker = CostTracker(budget=0.0000001, warn_only=True)
    with tracker, pytest.warns(BudgetWarning):
        tracker.check(1.0)


def test_status_and_reset(anthropic_response):
    tracker = CostTracker(budget=5.0)
    tracker.record_response(anthropic_response(1000, 500), "claude-sonnet-4-5")
    assert tracker.status().session_spend > 0
    tracker.reset()
    assert tracker.total_cost == 0.0
    assert tracker.status().session_spend == 0.0
    assert len(tracker) == 0


def test_limits_shorthand_conflict():
    from tokentab import BudgetLimits

    with pytest.raises(TypeError):
        CostTracker(budget=1.0, limits=BudgetLimits(per_session=2.0))


def test_repr_is_useful():
    tracker = CostTracker(budget=1.0, name="x")
    assert "CostTracker" in repr(tracker) and "x" in repr(tracker)
