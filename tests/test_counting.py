from __future__ import annotations

import pytest

from tokentab import (
    HeuristicCounter,
    count_message_tokens,
    count_tokens,
    get_counter,
    set_counter,
)


def test_heuristic_is_proportional():
    counter = HeuristicCounter()
    assert counter.count("") == 0
    assert counter.count("a") == 1
    short = counter.count("hello world")
    long = counter.count("hello world" * 100)
    assert long > short * 50


def test_count_tokens_empty():
    assert count_tokens("", "gpt-4o") == 0


def test_message_counting_accumulates():
    messages = [
        {"role": "user", "content": "Summarize the attached document."},
        {"role": "assistant", "content": "Sure, here is a summary."},
    ]
    total = count_message_tokens(messages, "claude-sonnet-4-5")
    single = count_message_tokens([messages[0]], "claude-sonnet-4-5")
    assert total > single > 0


def test_message_counting_handles_content_blocks():
    blocks = [{"role": "user", "content": [
        {"type": "text", "text": "first part"},
        {"type": "text", "text": "second part"},
    ]}]
    flat = [{"role": "user", "content": "first part\nsecond part"}]
    assert count_message_tokens(blocks, "gpt-4o") == count_message_tokens(flat, "gpt-4o")


def test_system_and_tools_are_counted():
    base = count_message_tokens([{"role": "user", "content": "hi"}], "gpt-4o")
    with_system = count_message_tokens(
        [{"role": "user", "content": "hi"}], "gpt-4o", system="You are a helpful assistant."
    )
    with_tools = count_message_tokens(
        [{"role": "user", "content": "hi"}], "gpt-4o",
        tools=[{"name": "search", "description": "search the web"}],
    )
    assert with_system > base
    assert with_tools > base


def test_bare_string_and_empty_input():
    assert count_message_tokens("just text", "gpt-4o") > 0
    assert count_message_tokens([], "gpt-4o") == 0


def test_custom_counter_override():
    class Fixed:
        def count(self, text: str, model: str) -> int:
            return 42

    set_counter("my-model", Fixed())
    try:
        assert get_counter("my-model").count("anything", "my-model") == 42
        assert count_tokens("anything", "my-model") == 42
    finally:
        set_counter("my-model", None)


def test_anthropic_counter_falls_back_on_error():
    from tokentab import AnthropicCounter

    class Boom:
        class messages:
            @staticmethod
            def count_tokens(**kwargs):
                raise RuntimeError("network down")

    counter = AnthropicCounter(client=Boom())
    # A counting failure must degrade to an estimate, never break the caller.
    assert counter.count("some text here", "claude-sonnet-4-5") > 0


def test_anthropic_counter_uses_endpoint():
    from tokentab import AnthropicCounter

    class Result:
        input_tokens = 123

    class Client:
        class messages:
            @staticmethod
            def count_tokens(**kwargs):
                return Result()

    counter = AnthropicCounter(client=Client())
    assert counter.count("hello", "claude-sonnet-4-5") == 123


@pytest.mark.skipif(
    pytest.importorskip("tiktoken", reason="tiktoken not installed") is None, reason=""
)
def test_tiktoken_counter_is_exact():
    from tokentab import TiktokenCounter

    counter = TiktokenCounter()
    assert counter.count("hello world", "gpt-4o") == 2
