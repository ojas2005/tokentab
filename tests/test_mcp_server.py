"""Tests for the MCP server. Skipped where the mcp extra is unavailable (Python 3.9)."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import math
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed (requires Python 3.10+)")

from tokentab import mcp_server as ms

EXPECTED_TOOLS = {
    "estimate_cost",
    "estimate_prompt_cost",
    "compare_models",
    "count_tokens",
    "plan_budget",
    "get_model_pricing",
    "list_models",
}


def _resolve(value):
    """Await the value if the SDK returned a coroutine, so tests work either way."""
    if inspect.isawaitable(value):
        async def _inner():
            return await value

        return asyncio.run(_inner())
    return value


def _field(obj, snake, camel):
    """Read a protocol field under its mcp 2.x (snake_case) or 1.x (camelCase) name."""
    value = getattr(obj, snake, None)
    return value if value is not None else getattr(obj, camel, None)


def _payload(result):
    """Pull the tool's dict back out of a CallToolResult."""
    for attr in ("structured_content", "structuredContent"):
        data = getattr(result, attr, None)
        if data:
            return data.get("result", data) if isinstance(data.get("result"), dict) else data
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------- registration


def test_every_tool_is_registered_and_read_only():
    tools = {t.name: t for t in _resolve(ms.server.list_tools())}
    assert set(tools) >= EXPECTED_TOOLS
    for name in EXPECTED_TOOLS:
        ann = tools[name].annotations
        assert ann is not None, name
        assert _field(ann, "read_only_hint", "readOnlyHint") is True, name
        assert _field(ann, "destructive_hint", "destructiveHint") is False, name
        assert _field(ann, "open_world_hint", "openWorldHint") is False, name
        assert tools[name].description, name


def test_tool_schemas_expose_arguments():
    tools = {t.name: t for t in _resolve(ms.server.list_tools())}
    schema = _field(tools["estimate_cost"], "input_schema", "inputSchema")
    assert {"model", "input_tokens"} <= set(schema["required"])
    assert "cache_read_tokens" in schema["properties"]


# ---------------------------------------------------------------- tool logic


def test_estimate_cost_matches_library():
    res = ms.estimate_cost("claude-sonnet-4-5", 1000, 500)
    assert res["total_cost_usd"] == pytest.approx(0.0105)
    assert res["breakdown_usd"]["input"] == pytest.approx(0.003)
    assert res["breakdown_usd"]["output"] == pytest.approx(0.0075)
    assert res["provider"] == "anthropic"
    assert res["pricing_as_of"]


def test_estimate_cost_prices_cache_separately():
    res = ms.estimate_cost("claude-sonnet-4-5", 0, 0, cache_read_tokens=100_000)
    assert res["total_cost_usd"] == pytest.approx(0.03)


def test_gateway_ids_resolve():
    res = ms.estimate_cost("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 1_000_000)
    assert res["priced_as"] == "claude-sonnet-4-5"
    assert res["total_cost_usd"] == pytest.approx(3.0)


def test_unknown_model_suggests_close_matches():
    with pytest.raises(ValueError, match=r"Did you mean: .*claude-sonnet-4-5"):
        ms.estimate_cost("claude-sonet-4-5", 10)


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        ms.estimate_cost("gpt-4o", -1)


def test_compare_models_is_sorted_cheapest_first():
    res = ms.compare_models(40_000, 2_000, limit=100)
    costs = [row["cost_usd"] for row in res["models"]]
    assert costs == sorted(costs)
    assert res["models"][0]["times_cheapest"] == 1.0
    assert not any(row["model"].startswith("text-embedding") for row in res["models"])


def test_compare_models_filters():
    res = ms.compare_models(1000, 100, provider="anthropic", limit=100)
    assert res["models"] and all(row["provider"] == "anthropic" for row in res["models"])
    res = ms.compare_models(1000, 100, models=["gpt-4o", "gpt-4o-mini"])
    assert [row["model"] for row in res["models"]] == ["gpt-4o-mini", "gpt-4o"]
    with pytest.raises(ValueError):
        ms.compare_models(1000, provider="nonexistent")


def test_count_tokens_reports_whether_exact():
    claude = ms.count_tokens("Hello there, how are you doing today?", "claude-sonnet-4-5")
    assert claude["tokens"] > 0
    assert claude["token_count_exact"] is False
    assert "API key" in claude["token_count_note"]

    gpt = ms.count_tokens("Hello there, how are you doing today?", "gpt-4o")
    has_tiktoken = importlib.util.find_spec("tiktoken") is not None
    assert gpt["token_count_exact"] is has_tiktoken


def test_estimate_prompt_cost_includes_output_allowance():
    without = ms.estimate_prompt_cost("Summarize the attached report.", "gpt-4o")
    with_output = ms.estimate_prompt_cost("Summarize the attached report.", "gpt-4o", 1000)
    assert with_output["total_cost_usd"] > without["total_cost_usd"]
    assert "output_note" in without and "output_note" not in with_output
    assert with_output["output_cost_usd"] == pytest.approx(0.01)


def test_plan_budget():
    res = ms.plan_budget("gpt-4o-mini", 1.0, 1000, 100)
    per_call = 1000 / 1e6 * 0.15 + 100 / 1e6 * 0.6
    assert res["cost_per_call_usd"] == pytest.approx(per_call)
    assert res["max_calls"] == math.floor(1.0 / per_call)
    assert res["cost_of_max_calls_usd"] <= 1.0
    with pytest.raises(ValueError):
        ms.plan_budget("gpt-4o", 0, 1000)


def test_get_model_pricing_and_list_models():
    res = ms.get_model_pricing("gpt-4o-2024-08-06")
    assert res["priced_as"] == "gpt-4o"
    assert res["rates"]["input"] == pytest.approx(2.5)
    google = ms.list_models(provider="google")
    assert google["count"] > 0 and all(m["provider"] == "google" for m in google["models"])
    haiku = ms.list_models(contains="haiku")
    assert haiku["count"] > 0 and all("haiku" in m["model"] for m in haiku["models"])
    assert google["pricing_as_of"]


# ---------------------------------------------------------------- through the server


def test_call_tool_through_the_server():
    result = _resolve(ms.server.call_tool("estimate_cost", {"model": "gpt-4o", "input_tokens": 1_000_000}))
    assert _payload(result)["total_cost_usd"] == pytest.approx(2.5)


def test_input_errors_reach_the_client_with_their_message():
    """An unknown model or bad argument must come back as a ToolError carrying
    its message. Anything else is treated by MCP as a crash: the client sees
    only "Error executing tool" and the "Did you mean" hint is lost."""
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    cases = [
        ("estimate_cost", {"model": "claude-sonet-4-5", "input_tokens": 10},
         r"Did you mean: .*claude-sonnet-4-5"),
        ("estimate_cost", {"model": "gpt-4o", "input_tokens": -5}, "non-negative"),
        ("plan_budget", {"model": "gpt-4o", "budget_usd": 0, "input_tokens": 10}, "budget_usd"),
        ("compare_models", {"input_tokens": 10, "provider": "nope"}, "No models matched"),
    ]
    for tool, args, message in cases:
        with pytest.raises(ToolError, match=message) as info:
            _resolve(ms.server.call_tool(tool, args))
        assert not isinstance(info.value, UnexpectedToolError), (tool, args)


def test_wrapping_keeps_the_tool_schemas():
    tools = {t.name: t for t in _resolve(ms.server.list_tools())}
    props = _field(tools["plan_budget"], "input_schema", "inputSchema")["properties"]
    assert {"model", "budget_usd", "input_tokens", "output_tokens"} <= set(props)
    assert "how many calls" in tools["plan_budget"].description.lower()


def test_stdio_end_to_end():
    """Spawn the real server process and talk to it the way Claude Code does."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def session_round_trip():
        # The MCP client launches servers with a sanitized environment, so point
        # the child at the same tokentab this test imported (src/ or installed).
        pkg_root = str(Path(ms.__file__).resolve().parent.parent)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            p for p in (pkg_root, os.environ.get("PYTHONPATH", "")) if p
        )}
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "tokentab.mcp_server"], env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                called = await session.call_tool(
                    "estimate_cost",
                    {"model": "claude-sonnet-4-5", "input_tokens": 1000, "output_tokens": 500},
                )
                bad = await session.call_tool(
                    "estimate_cost", {"model": "claude-sonet-4-5", "input_tokens": 10}
                )
                return listed, called, bad

    listed, called, bad = asyncio.run(asyncio.wait_for(session_round_trip(), timeout=60))
    assert {t.name for t in listed.tools} >= EXPECTED_TOOLS
    assert _payload(called)["total_cost_usd"] == pytest.approx(0.0105)
    # Over the wire, the typo comes back as an error result the model can act on.
    assert _field(bad, "is_error", "isError") is True
    assert "Did you mean: claude-sonnet-4-5" in bad.content[0].text
