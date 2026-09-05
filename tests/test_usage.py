from __future__ import annotations

import pytest

from tokentab import TokenUsage, extract_usage


def test_anthropic_shape(anthropic_response):
    usage = extract_usage(anthropic_response(i=1000, o=500, cr=2000, cw=300))
    assert usage == TokenUsage(
        input_tokens=1000, output_tokens=500, cache_read_tokens=2000, cache_write_tokens=300
    )


def test_anthropic_input_tokens_exclude_cache():
    """Anthropic reports input_tokens exclusive of cache; nothing is subtracted."""
    usage = extract_usage(
        {"usage": {"input_tokens": 10, "output_tokens": 5,
                   "cache_read_input_tokens": 4000, "cache_creation_input_tokens": 100}}
    )
    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.cache_read_tokens == 4000


def test_openai_input_tokens_include_cache(openai_response):
    """OpenAI's prompt_tokens is inclusive, so cached tokens must be subtracted once."""
    usage = extract_usage(openai_response(p=1200, c=50, cached=1000))
    assert usage is not None
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 1000
    assert usage.output_tokens == 50


def test_gemini_shape():
    usage = extract_usage(
        {"usage_metadata": {"prompt_token_count": 300, "candidates_token_count": 40,
                            "cached_content_token_count": 100, "total_token_count": 340}}
    )
    assert usage is not None
    assert usage.output_tokens == 40
    assert usage.cache_read_tokens == 100


def test_langchain_usage_metadata():
    usage = extract_usage(
        {"usage_metadata": {"input_tokens": 10, "output_tokens": 20,
                            "input_token_details": {"cache_read": 5}}}
    )
    assert usage is not None
    assert usage.cache_read_tokens == 5


def test_reasoning_tokens():
    usage = extract_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 500,
                   "completion_tokens_details": {"reasoning_tokens": 450}}}
    )
    assert usage is not None
    assert usage.reasoning_tokens == 450
    # Reasoning tokens are already inside output_tokens; don't double count.
    assert usage.total_tokens == 510


def test_nested_llm_output():
    usage = extract_usage({"llm_output": {"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}}})
    assert usage == TokenUsage(input_tokens=7, output_tokens=3)


def test_no_usage_returns_none():
    assert extract_usage(None) is None
    assert extract_usage("a plain string") is None
    assert extract_usage({"choices": []}) is None
    assert extract_usage({"usage": {"input_tokens": 0, "output_tokens": 0}}) is None


def test_usage_arithmetic():
    a = TokenUsage(input_tokens=1, output_tokens=2, cache_read_tokens=3)
    b = TokenUsage(input_tokens=10, output_tokens=20, cache_write_tokens=5)
    total = a + b
    assert total.input_tokens == 11
    assert total.output_tokens == 22
    assert total.cache_read_tokens == 3
    assert total.cache_write_tokens == 5
    assert total.total_tokens == 41


def test_usage_validation():
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=-1)
    with pytest.raises(TypeError):
        TokenUsage(input_tokens="ten")  # type: ignore[arg-type]
