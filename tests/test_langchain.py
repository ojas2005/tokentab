from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tokentab import BudgetExceededError, CostTracker, TokenTabCallbackHandler


def _serialized(name="ChatOpenAI"):
    return {"id": ["langchain", "chat_models", name], "name": name}


def _result_with_usage(input_tokens=1000, output_tokens=500, cache_read=0):
    message = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_handler_records_chat_model_call():
    handler = TokenTabCallbackHandler(budget=5.0, tag="agent")
    run_id = uuid4()
    handler.on_chat_model_start(
        _serialized(),
        [[HumanMessage(content="summarize this document")]],
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    handler.on_llm_end(_result_with_usage(1000, 500), run_id=run_id)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.model == "gpt-4o"
    assert record.tag == "agent"
    assert record.estimated is False
    assert record.cost == pytest.approx(1000 / 1e6 * 2.5 + 500 / 1e6 * 10)
    assert record.duration_s is not None


def test_handler_blocks_over_budget_before_the_call():
    handler = TokenTabCallbackHandler(per_request=0.000001)
    with pytest.raises(BudgetExceededError):
        handler.on_chat_model_start(
            _serialized(),
            [[HumanMessage(content="x" * 100_000)]],
            run_id=uuid4(),
            invocation_params={"model": "claude-opus-4-1"},
        )
    assert handler.records == []


def test_handler_reads_model_from_metadata():
    handler = TokenTabCallbackHandler(budget=1.0)
    run_id = uuid4()
    handler.on_chat_model_start(
        _serialized(), [[HumanMessage(content="hi")]], run_id=run_id,
        metadata={"ls_model_name": "claude-sonnet-4-5"},
    )
    handler.on_llm_end(_result_with_usage(100, 50), run_id=run_id)
    assert handler.records[0].model == "claude-sonnet-4-5"


def test_handler_reads_cache_tokens():
    handler = TokenTabCallbackHandler(budget=1.0)
    run_id = uuid4()
    handler.on_chat_model_start(
        _serialized(), [[HumanMessage(content="hi")]], run_id=run_id,
        invocation_params={"model": "claude-sonnet-4-5"},
    )
    handler.on_llm_end(_result_with_usage(100, 50, cache_read=4000), run_id=run_id)
    assert handler.records[0].usage.cache_read_tokens == 4000


def test_handler_completion_style_llm_start():
    handler = TokenTabCallbackHandler(budget=1.0)
    run_id = uuid4()
    handler.on_llm_start(
        _serialized("OpenAI"), ["a plain prompt"], run_id=run_id,
        invocation_params={"model_name": "gpt-3.5-turbo"},
    )
    result = LLMResult(
        generations=[[]],
        llm_output={"token_usage": {"prompt_tokens": 20, "completion_tokens": 10}},
    )
    handler.on_llm_end(result, run_id=run_id)
    assert handler.records[0].model == "gpt-3.5-turbo"
    assert handler.records[0].input_tokens == 20


def test_handler_falls_back_to_estimate_without_usage():
    handler = TokenTabCallbackHandler(budget=1.0)
    run_id = uuid4()
    handler.on_chat_model_start(
        _serialized(), [[HumanMessage(content="hello world")]], run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    handler.on_llm_end(LLMResult(generations=[[]]), run_id=run_id)
    assert handler.records[0].estimated is True


def test_handler_drops_failed_runs():
    handler = TokenTabCallbackHandler(budget=1.0)
    run_id = uuid4()
    handler.on_chat_model_start(
        _serialized(), [[HumanMessage(content="hi")]], run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    handler.on_llm_error(RuntimeError("provider down"), run_id=run_id)
    handler.on_llm_end(_result_with_usage(), run_id=run_id)  # late/duplicate end
    assert handler.records == []


def test_handler_shares_an_active_tracker():
    """A handler with no budget of its own joins the enclosing CostTracker."""
    with CostTracker(budget=5.0, tag="run") as tracker:
        handler = TokenTabCallbackHandler()
        run_id = uuid4()
        handler.on_chat_model_start(
            _serialized(), [[HumanMessage(content="hi")]], run_id=run_id,
            invocation_params={"model": "gpt-4o"},
        )
        handler.on_llm_end(_result_with_usage(100, 50), run_id=run_id)
    assert len(tracker.records) == 1
    assert tracker.total_cost > 0
    assert handler.total_cost == pytest.approx(tracker.total_cost)


def test_handler_enforces_shared_budget():
    with CostTracker(budget=0.0000001) as tracker:
        handler = TokenTabCallbackHandler()
        with pytest.raises(BudgetExceededError):
            handler.on_chat_model_start(
                _serialized(), [[HumanMessage(content="x" * 10_000)]], run_id=uuid4(),
                invocation_params={"model": "gpt-4o"},
            )
    assert tracker.records == []


def test_handler_report():
    handler = TokenTabCallbackHandler(budget=5.0, tag="agent")
    for _ in range(3):
        run_id = uuid4()
        handler.on_chat_model_start(
            _serialized(), [[HumanMessage(content="hi")]], run_id=run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_end(_result_with_usage(100, 20), run_id=run_id)
    report = handler.report()
    assert report.call_count == 3
    assert report.by_model["gpt-4o-mini"].calls == 3
    assert "by model" in report.summary()


def test_raise_error_flag_is_set():
    """LangChain only propagates handler exceptions when raise_error is true;
    without it BudgetExceededError would be logged and the chain would continue."""
    assert TokenTabCallbackHandler.raise_error is True


def test_output_allowance_uses_max_tokens():
    """Output costs more than input; budgeting from the prompt alone lets a
    session overshoot by a whole response, so max_tokens is the better default."""
    handler = TokenTabCallbackHandler(budget=100.0)
    assert handler._output_allowance({"invocation_params": {"max_tokens": 4096}}) == 4096
    assert handler._output_allowance({"invocation_params": {"max_output_tokens": 512}}) == 512
    assert handler._output_allowance({}) == 0

    explicit = TokenTabCallbackHandler(budget=100.0, expected_output_tokens=1000)
    assert explicit._output_allowance({"invocation_params": {"max_tokens": 4096}}) == 1000


def test_max_tokens_makes_the_guard_stop_sooner():
    handler = TokenTabCallbackHandler(per_request=0.02)
    # Prompt alone is cheap; prompt + 4096 output tokens on Sonnet is not.
    with pytest.raises(BudgetExceededError):
        handler.on_chat_model_start(
            _serialized(), [[HumanMessage(content="short prompt")]], run_id=uuid4(),
            invocation_params={"model": "claude-sonnet-4-5", "max_tokens": 4096},
        )
    # Without max_tokens the same prompt sails through.
    handler.on_chat_model_start(
        _serialized(), [[HumanMessage(content="short prompt")]], run_id=uuid4(),
        invocation_params={"model": "claude-sonnet-4-5"},
    )
