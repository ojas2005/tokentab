"""End-to-end tests against a real LangChain runnable.

test_langchain.py drives the callbacks by hand, which verifies tokentab's own
logic but not that LangChain actually invokes them. These tests go through real
`invoke`/`ainvoke` calls, which is what catches wiring problems: whether the
handler is called at all, whether BudgetExceededError escapes the chain rather
than being logged and swallowed, and what LangChain passes for tags and model.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tokentab import (
    BudgetExceededError,
    CostTracker,
    PricingRegistry,
    TokenTabCallbackHandler,
)


@pytest.fixture
def llm():
    return FakeListChatModel(responses=["a response"])


@pytest.fixture
def priced_registry():
    """The fake model needs real-money pricing before a budget can be breached."""
    registry = PricingRegistry()
    registry.register("FakeListChatModel", input=3.0, output=15.0)
    return registry


def _handler(registry, **kwargs):
    tracker = CostTracker(registry=registry, ambient=False, **kwargs)
    return TokenTabCallbackHandler(tracker)


def test_langchain_actually_invokes_the_handler(llm, priced_registry):
    handler = _handler(priced_registry, budget=5.0)
    llm.invoke("hello world", config={"callbacks": [handler]})
    assert len(handler.records) == 1
    assert handler.total_cost > 0


def test_handler_works_on_async_runs(llm, priced_registry):
    """LangChain dispatches sync handlers on async runs from a worker thread,
    where context variables do not reach -- the ambient tracker stack does."""
    handler = _handler(priced_registry, budget=5.0)
    asyncio.run(llm.ainvoke("hello world", config={"callbacks": [handler]}))
    assert len(handler.records) == 1


def test_budget_error_escapes_invoke(llm, priced_registry):
    """raise_error=True is what makes this propagate instead of being logged."""
    handler = _handler(priced_registry, per_request=0.0000001)
    with pytest.raises(BudgetExceededError):
        llm.invoke("x" * 5000, config={"callbacks": [handler]})


def test_budget_error_escapes_ainvoke(llm, priced_registry):
    handler = _handler(priced_registry, per_request=0.0000001)
    with pytest.raises(BudgetExceededError):
        asyncio.run(llm.ainvoke("x" * 5000, config={"callbacks": [handler]}))


def test_budget_caps_a_multi_step_chain(llm, priced_registry):
    chain = ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()
    handler = _handler(priced_registry, budget=0.001)
    completed = 0
    with pytest.raises(BudgetExceededError):
        for _ in range(20):
            chain.invoke({"q": "detail " * 100}, config={"callbacks": [handler]})
            completed += 1
    assert 0 < completed < 20
    # The run stopped before the limit, not after it.
    assert handler.total_cost <= 0.001


def test_chain_records_are_not_labeled_with_internal_tags(llm, priced_registry):
    """Inside a chain LangChain passes tags like "seq:step:2"; an explicit
    handler tag must survive that."""
    chain = ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()
    tracker = CostTracker(registry=priced_registry, budget=5.0, ambient=False)
    handler = TokenTabCallbackHandler(tracker, tag="my-pipeline")
    chain.invoke({"q": "hello"}, config={"callbacks": [handler]})
    assert handler.records[0].tag == "my-pipeline"
    assert all(":" not in (r.tag or "") for r in handler.records)
