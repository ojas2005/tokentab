"""MCP server that exposes tokentab's pricing and token counting as tools.

Register it with Claude Code (or any MCP client) and ask cost questions in plain
language -- "what would this prompt cost on Opus?", "which model is cheapest for
40k in / 2k out?", "how many calls fit in $5?"::

    pip install 'tokentab[mcp]'
    claude mcp add -s user tokentab -- tokentab-mcp

Every tool is read-only and runs locally. Nothing here calls a provider or needs
an API key: prices come from the bundled table (or ``TOKENTAB_PRICING_FILE``),
and token counts from tiktoken when installed, a character heuristic otherwise.

Needs the ``mcp`` extra, which requires Python 3.10+. The rest of tokentab still
supports 3.9, so this module is never imported by ``tokentab`` itself.
"""

from __future__ import annotations

import difflib
import functools
import math
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from . import __version__
from .counting import count_message_tokens, get_counter
from .exceptions import TokenTabError
from .pricing import ModelPricing, default_registry, detect_provider
from .usage import TokenUsage

try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The tokentab MCP server needs the mcp extra, which requires Python 3.10+: "
        "pip install 'tokentab[mcp]'"
    ) from exc

__all__ = [
    "server",
    "main",
    "estimate_cost",
    "estimate_prompt_cost",
    "compare_models",
    "count_tokens",
    "plan_budget",
    "get_model_pricing",
    "list_models",
]

INSTRUCTIONS = """\
tokentab prices LLM API calls. Use these tools when the user asks what a model,
call or prompt would cost, which model is cheapest for a workload, how many
tokens some text is, or how many calls fit in a budget.

Prices are USD per million tokens from a bundled table of public list prices;
every result carries the table's `pricing_as_of` date. They can be out of date
and do not reflect negotiated rates, batch discounts or long-context tiers, so
say so when precision matters. Token counts are exact for OpenAI models when
tiktoken is installed and estimates otherwise; each result says which.
"""

# Every tool only reads local data: no side effects, same answer every time,
# nothing outside this process is contacted.
_READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)

server = MCPServer(name="tokentab", instructions=INSTRUCTIONS, version=__version__)


# ---------------------------------------------------------------- helpers


def _money(value: float) -> float:
    return round(value, 6)


def _as_of() -> Optional[str]:
    value = default_registry().meta.get("as_of")
    return str(value) if value is not None else None


def _pricing(model: str) -> ModelPricing:
    """Look up a model, turning a miss into an error that names close matches."""
    registry = default_registry()
    found = registry.get(model)
    if found is not None:
        return found
    close = difflib.get_close_matches(
        model.strip().lower(), list(registry.models()), n=3, cutoff=0.5
    )
    if close:
        hint = f" Did you mean: {', '.join(close)}?"
    else:
        hint = " Call list_models to see known models."
    raise ValueError(f"No pricing for model {model!r}.{hint}")


def _rates(pricing: ModelPricing) -> Dict[str, Any]:
    return {
        "input": pricing.input,
        "output": pricing.output,
        "cache_read": pricing.effective_cache_read,
        "cache_write": pricing.effective_cache_write,
        "unit": "USD per 1M tokens",
    }


def _counter_info(model: str) -> Dict[str, Any]:
    counter = get_counter(model)
    exact = bool(getattr(counter, "exact", False))
    info: Dict[str, Any] = {
        "token_count_method": type(counter).__name__,
        "token_count_exact": exact,
    }
    if not exact:
        info["token_count_note"] = (
            "Estimated at ~4 characters per token, typically within 10-20% for prose. "
            + (
                "Exact Claude counts need Anthropic's count_tokens endpoint, which requires an "
                "API key; this server does not use one."
                if detect_provider(model) == "anthropic"
                else "Install tiktoken for exact OpenAI counts."
            )
        )
    return info


# ------------------------------------------------------------------ tools


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Dict[str, Any]:
    """Price one LLM call from its token counts.

    Args:
        model: Model id, e.g. "claude-sonnet-4-5", "gpt-4o", "gemini-2.5-flash".
            Gateway and cloud ids such as "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            or "openai/gpt-4o" resolve too.
        input_tokens: Prompt tokens billed at the full input rate.
        output_tokens: Completion tokens.
        cache_read_tokens: Prompt tokens served from the provider's prompt cache.
        cache_write_tokens: Prompt tokens written to the cache (Anthropic bills a premium).
    """
    pricing = _pricing(model)
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    cost = pricing.cost(usage)
    return {
        "model": model,
        "priced_as": pricing.model,
        "provider": pricing.provider or detect_provider(model),
        "total_cost_usd": _money(cost.total),
        "breakdown_usd": {
            "input": _money(cost.input_cost),
            "output": _money(cost.output_cost),
            "cache_read": _money(cost.cache_read_cost),
            "cache_write": _money(cost.cache_write_cost),
        },
        "rates": _rates(pricing),
        "pricing_as_of": _as_of(),
    }


def estimate_prompt_cost(
    prompt: str,
    model: str,
    expected_output_tokens: int = 0,
    system: Optional[str] = None,
) -> Dict[str, Any]:
    """Count a prompt's tokens and price it, as tokentab's budget guard does before a call.

    Args:
        prompt: The user message text to price.
        model: Model id, e.g. "claude-opus-4-1" or "gpt-4o-mini".
        expected_output_tokens: Allowance for the response, e.g. your max_tokens.
            Output usually costs 3-5x input, so leaving this at 0 under-counts.
        system: Optional system prompt, counted as part of the input.
    """
    pricing = _pricing(model)
    input_tokens = count_message_tokens(
        [{"role": "user", "content": prompt}], model, counter=get_counter(model), system=system
    )
    cost = pricing.cost(TokenUsage(input_tokens=input_tokens, output_tokens=expected_output_tokens))
    result: Dict[str, Any] = {
        "model": model,
        "priced_as": pricing.model,
        "input_tokens": input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "input_cost_usd": _money(cost.input_cost),
        "output_cost_usd": _money(cost.output_cost),
        "total_cost_usd": _money(cost.total),
        "pricing_as_of": _as_of(),
    }
    result.update(_counter_info(model))
    if expected_output_tokens == 0:
        result["output_note"] = (
            "Response cost not included. Pass expected_output_tokens (e.g. your max_tokens) "
            "for a realistic total."
        )
    return result


def compare_models(
    input_tokens: int,
    output_tokens: int = 0,
    models: Optional[List[str]] = None,
    provider: Optional[str] = None,
    cache_read_tokens: int = 0,
    limit: int = 15,
) -> Dict[str, Any]:
    """Price the same workload across models, cheapest first.

    Args:
        input_tokens: Prompt tokens per call.
        output_tokens: Completion tokens per call.
        models: Specific model ids to compare. Omit to compare every known chat model.
        provider: Restrict to one provider: "anthropic", "openai" or "google".
        cache_read_tokens: Prompt tokens served from cache, per call.
        limit: Maximum number of rows to return.
    """
    usage = TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read_tokens
    )
    if models:
        candidates = [(name, _pricing(name)) for name in models]
    else:
        # Embedding models have no output price and would top every chat comparison.
        candidates = [
            (name, pricing)
            for name, pricing in default_registry().models().items()
            if not name.startswith("text-embedding")
        ]
    wanted = provider.strip().lower() if provider else None
    rows: List[Dict[str, Any]] = []
    for name, pricing in candidates:
        prov = pricing.provider or detect_provider(name)
        if wanted and prov != wanted:
            continue
        rows.append(
            {"model": name, "provider": prov, "cost_usd": _money(pricing.cost(usage).total)}
        )
    if not rows:
        raise ValueError(
            f"No models matched provider={provider!r}. Use anthropic, openai or google."
        )
    rows.sort(key=lambda row: (row["cost_usd"], row["model"]))
    cheapest = rows[0]["cost_usd"]
    for row in rows:
        row["times_cheapest"] = round(row["cost_usd"] / cheapest, 1) if cheapest > 0 else None
    return {
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
        },
        "models": rows[: max(limit, 1)],
        "models_compared": len(rows),
        "pricing_as_of": _as_of(),
    }


def count_tokens(text: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Count the tokens in a piece of text for a given model.

    Args:
        text: The text to count.
        model: Model whose tokenizer to use. Exact for OpenAI models when tiktoken
            is installed; an estimate otherwise.
    """
    result: Dict[str, Any] = {"model": model, "tokens": get_counter(model).count(text, model)}
    result.update(_counter_info(model))
    return result


def plan_budget(
    model: str,
    budget_usd: float,
    input_tokens: int,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Dict[str, Any]:
    """Work out how many calls of a given size fit in a budget.

    Args:
        model: Model id.
        budget_usd: The budget in US dollars.
        input_tokens: Prompt tokens per call.
        output_tokens: Completion tokens per call.
        cache_read_tokens: Prompt tokens served from cache, per call.
    """
    if budget_usd <= 0:
        raise ValueError("budget_usd must be greater than 0")
    pricing = _pricing(model)
    per_call = pricing.cost(
        TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )
    ).total
    if per_call <= 0:
        return {"model": model, "cost_per_call_usd": 0.0, "max_calls": None,
                "note": "These token counts cost nothing at this model's rates."}
    max_calls = math.floor(budget_usd / per_call)
    return {
        "model": model,
        "priced_as": pricing.model,
        "budget_usd": budget_usd,
        "cost_per_call_usd": _money(per_call),
        "max_calls": max_calls,
        "cost_of_max_calls_usd": _money(max_calls * per_call),
        "pricing_as_of": _as_of(),
    }


def get_model_pricing(model: str) -> Dict[str, Any]:
    """Show the rates tokentab uses for a model, and which table entry it resolved to.

    Args:
        model: Model id; dated snapshots and gateway ids resolve to their base model.
    """
    pricing = _pricing(model)
    return {
        "model": model,
        "priced_as": pricing.model,
        "provider": pricing.provider or detect_provider(model),
        "rates": _rates(pricing),
        "pricing_as_of": _as_of(),
    }


def list_models(provider: Optional[str] = None, contains: Optional[str] = None) -> Dict[str, Any]:
    """List the models tokentab has prices for.

    Args:
        provider: Only this provider: "anthropic", "openai" or "google".
        contains: Only model ids containing this text, e.g. "haiku" or "mini".
    """
    wanted = provider.strip().lower() if provider else None
    needle = contains.strip().lower() if contains else None
    rows = []
    for name, pricing in sorted(default_registry().models().items()):
        prov = pricing.provider or detect_provider(name)
        if (wanted and prov != wanted) or (needle and needle not in name):
            continue
        rows.append(
            {"model": name, "provider": prov, "input": pricing.input, "output": pricing.output}
        )
    meta = default_registry().meta
    return {
        "models": rows,
        "count": len(rows),
        "unit": "USD per 1M tokens",
        "pricing_as_of": meta.get("as_of"),
        "pricing_note": meta.get("note"),
    }


_Tool = TypeVar("_Tool", bound=Callable[..., Dict[str, Any]])


def _user_errors(fn: _Tool) -> _Tool:
    """Report bad input to the model instead of crashing the call.

    MCP treats any exception other than ``ToolError`` as a server crash: the
    client gets a generic "Error executing tool" and the real message stays on
    stderr. Input problems -- an unknown model, negative token counts, a zero
    budget, an unreadable pricing file -- are the caller's to fix, so their
    message (including the "Did you mean" hint) has to reach the model.
    Called directly from Python, the functions still raise ``ValueError``.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except (ValueError, TypeError, TokenTabError) as exc:
            raise ToolError(str(exc)) from exc

    return cast(_Tool, wrapper)


for _tool in (
    estimate_cost,
    estimate_prompt_cost,
    compare_models,
    count_tokens,
    plan_budget,
    get_model_pricing,
    list_models,
):
    server.tool(annotations=_READ_ONLY)(_user_errors(_tool))


def main() -> None:
    """Console entry point: serve over stdio."""
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
