from __future__ import annotations

import pytest

from tokentab import PricingRegistry, set_default_tracker
from tokentab.tracker import _ambient


@pytest.fixture(autouse=True)
def _clean_global_state():
    """Keep the process-wide tracker/ambient stack from leaking between tests."""
    _ambient.clear()
    set_default_tracker(None)
    yield
    _ambient.clear()
    set_default_tracker(None)


@pytest.fixture
def registry() -> PricingRegistry:
    return PricingRegistry()


class FakeAnthropicResponse:
    """Shape of anthropic.types.Message."""

    class Usage:
        def __init__(self, i, o, cr=0, cw=0):
            self.input_tokens = i
            self.output_tokens = o
            self.cache_read_input_tokens = cr
            self.cache_creation_input_tokens = cw

    def __init__(self, i=1000, o=500, cr=0, cw=0):
        self.usage = self.Usage(i, o, cr, cw)


class FakeOpenAIResponse:
    """Shape of openai.types.chat.ChatCompletion."""

    class Details:
        def __init__(self, cached):
            self.cached_tokens = cached

    class Usage:
        def __init__(self, p, c, cached=0):
            self.prompt_tokens = p
            self.completion_tokens = c
            self.total_tokens = p + c
            self.prompt_tokens_details = FakeOpenAIResponse.Details(cached)

    def __init__(self, p=1000, c=500, cached=0):
        self.usage = self.Usage(p, c, cached)


@pytest.fixture
def anthropic_response():
    return FakeAnthropicResponse


@pytest.fixture
def openai_response():
    return FakeOpenAIResponse
