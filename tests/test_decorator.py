from __future__ import annotations

import asyncio
import warnings

import pytest

from tokentab import BudgetExceededError, CostTracker, default_tracker, track_cost


def test_decorator_records_actual_usage(openai_response):
    @track_cost(model="gpt-4o", tag="summarize")
    def call(messages):
        return openai_response(p=1000, c=500)

    with CostTracker(budget=5.0) as tracker:
        call([{"role": "user", "content": "hello"}])
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.model == "gpt-4o"
    assert record.tag == "summarize"
    assert record.estimated is False
    assert record.cost == pytest.approx(1000 / 1e6 * 2.5 + 500 / 1e6 * 10)


def test_decorator_blocks_before_spending():
    calls = []

    @track_cost(model="claude-opus-4-1")
    def expensive(messages):
        calls.append(1)
        return None

    with CostTracker(per_request=0.0001):
        with pytest.raises(BudgetExceededError):
            expensive("x" * 500_000)
    # The point of pre-call estimation: the function never ran.
    assert calls == []


def test_decorator_never_swallows_exceptions():
    @track_cost(model="gpt-4o")
    def failing(messages):
        raise ValueError("provider rejected the request")

    with CostTracker() as tracker:
        with pytest.raises(ValueError, match="provider rejected"):
            failing("hi")
    assert tracker.records == []


def test_decorator_preserves_return_value_and_metadata(openai_response):
    sentinel = openai_response()

    @track_cost(model="gpt-4o")
    def call(messages):
        """Docstring preserved."""
        return sentinel

    with CostTracker():
        assert call("hi") is sentinel
    assert call.__doc__ == "Docstring preserved."
    assert call.__name__ == "call"


def test_decorator_reads_model_from_arguments(anthropic_response):
    @track_cost()
    def call(model, messages):
        return anthropic_response(100, 50)

    with CostTracker() as tracker:
        call("claude-sonnet-4-5", "hello")
    assert tracker.records[0].model == "claude-sonnet-4-5"


def test_decorator_bare_form(openai_response):
    @track_cost
    def call(model, messages):
        return openai_response(p=10, c=5)

    with CostTracker() as tracker:
        call("gpt-4o", "hi")
    assert len(tracker.records) == 1


def test_decorator_uses_default_tracker_outside_a_block(openai_response):
    @track_cost(model="gpt-4o")
    def call(messages):
        return openai_response(p=100, c=10)

    call("hi")
    assert len(default_tracker().records) == 1


def test_decorator_explicit_tracker(openai_response):
    tracker = CostTracker(budget=1.0)

    @track_cost(model="gpt-4o", tracker=tracker)
    def call(messages):
        return openai_response(p=100, c=10)

    call("hi")
    assert len(tracker.records) == 1


def test_decorator_enforce_false_observes_only():
    @track_cost(model="claude-opus-4-1", enforce=False)
    def call(messages):
        return None

    with CostTracker(per_request=0.0000001):
        call("x" * 100_000)  # no raise


def test_decorator_records_estimate_when_response_has_no_usage():
    @track_cost(model="gpt-4o")
    def call(messages):
        return "plain string"

    with CostTracker() as tracker:
        call("hello world")
    assert tracker.records[0].estimated is True


def test_decorator_can_skip_estimate_records():
    @track_cost(model="gpt-4o", record_estimate=False)
    def call(messages):
        return "plain string"

    with CostTracker() as tracker:
        call("hello world")
    assert tracker.records == []


def test_decorator_custom_token_estimator():
    seen = {}

    @track_cost(model="gpt-4o", estimate_input_tokens=lambda payload: len(payload) * 10)
    def call(payload):
        seen["ran"] = True
        return "no usage"

    with CostTracker() as tracker:
        call([1, 2, 3])
    assert seen["ran"]
    assert tracker.records[0].input_tokens == 30


def test_decorator_on_async_function(openai_response):
    @track_cost(model="gpt-4o", tag="async")
    async def call(messages):
        await asyncio.sleep(0)
        return openai_response(p=100, c=20)

    with CostTracker() as tracker:
        asyncio.run(call("hi"))
    assert tracker.records[0].tag == "async"
    assert tracker.records[0].input_tokens == 100


def test_async_decorator_propagates_exceptions():
    @track_cost(model="gpt-4o")
    async def failing(messages):
        raise RuntimeError("boom")

    loop = asyncio.new_event_loop()
    try:
        with CostTracker():
            with pytest.raises(RuntimeError, match="boom"):
                loop.run_until_complete(failing("hi"))
    finally:
        loop.close()


def test_bookkeeping_error_does_not_destroy_result(openai_response):
    """A pricing failure *after* a paid-for call must not lose the response.

    ``payload`` is not a recognized prompt parameter, so there is no pre-call
    estimate and the failure can only happen on the way out.
    """
    sentinel = openai_response(p=10, c=5)

    @track_cost(model="totally-unknown-model", tracker=CostTracker(strict_pricing=True))
    def call(payload):
        return sentinel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert call(object()) is sentinel
    assert any("could not record usage" in str(w.message) for w in caught)


def test_strict_pricing_refuses_before_the_call():
    """Pre-call, an unknown model under strict_pricing is a hard stop: better to
    fail before the money leaves than to bill an untracked call."""
    from tokentab import UnknownModelError

    ran = []

    @track_cost(model="totally-unknown-model", tracker=CostTracker(strict_pricing=True))
    def call(messages):
        ran.append(1)
        return None

    with pytest.raises(UnknownModelError):
        call("hello")
    assert ran == []


def test_decorator_without_model_is_a_noop(openai_response):
    @track_cost()
    def call(messages):
        return openai_response()

    with CostTracker() as tracker:
        call("hi")
    assert tracker.records == []
