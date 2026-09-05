"""Runnable tour of tokentab, with a fake provider client.

    python examples/quickstart.py

No API key and no network access: FakeClient stands in for a real SDK and
returns responses in the exact shape Anthropic and OpenAI use.
"""

from __future__ import annotations

from tokentab import (
    BudgetExceededError,
    CostTracker,
    TokenUsage,
    estimate_cost,
    register_model,
    track_cost,
)


class FakeAnthropicClient:
    """Returns an object shaped like anthropic.types.Message."""

    class _Usage:
        def __init__(self, i: int, o: int, cache_read: int) -> None:
            self.input_tokens = i
            self.output_tokens = o
            self.cache_read_input_tokens = cache_read
            self.cache_creation_input_tokens = 0

    class _Message:
        def __init__(self, usage: FakeAnthropicClient._Usage) -> None:
            self.usage = usage
            self.content = [{"type": "text", "text": "...a helpful answer..."}]

    class messages:
        @staticmethod
        def create(model: str, messages: list, **kwargs: object) -> FakeAnthropicClient._Message:
            return FakeAnthropicClient._Message(FakeAnthropicClient._Usage(1200, 450, 8000))


client = FakeAnthropicClient()
MESSAGES = [{"role": "user", "content": "Summarize the Q3 earnings call transcript."}]


def main() -> None:
    print("=" * 68)
    print("1. What will this cost before I send it?")
    print("=" * 68)
    print(f"  1k in / 500 out on Sonnet: ${estimate_cost('claude-sonnet-4-5', 1000, 500):.6f}")
    print(f"  same call on Opus 4.1:     ${estimate_cost('claude-opus-4-1', 1000, 500):.6f}")
    print(f"  8k cached read on Sonnet:  "
          f"${estimate_cost('claude-sonnet-4-5', 0, 0, cache_read_tokens=8000):.6f}")

    print()
    print("=" * 68)
    print("2. Context manager: track a session against a budget")
    print("=" * 68)
    with CostTracker(budget=5.00, per_request=1.00, name="demo") as tracker:
        response = client.messages.create(model="claude-sonnet-4-5", messages=MESSAGES)
        tracker.record_response(response, "claude-sonnet-4-5", tag="summarize")
        print(f"  spent ${tracker.total_cost:.6f}, "
              f"${tracker.status().session_remaining:.6f} left of the session budget")

        print()
        print("  Guarding a call end to end (budget checked *before* it is sent):")
        with tracker.call(
            "claude-sonnet-4-5", messages=MESSAGES, expected_output_tokens=1000, tag="guarded"
        ) as call:
            print(f"    estimated first: ${call.estimated_cost:.6f}")
            call.set_response(client.messages.create(model="claude-sonnet-4-5", messages=MESSAGES))
        assert call.record is not None
        print(f"    actual recorded: ${call.record.cost:.6f}")

    print()
    print("=" * 68)
    print("3. Decorator")
    print("=" * 68)

    @track_cost(model="claude-sonnet-4-5", tag="decorated")
    def summarize(messages: list) -> object:
        return client.messages.create(model="claude-sonnet-4-5", messages=messages)

    with CostTracker(budget=5.00) as tracker:
        summarize(MESSAGES)
        print(f"  recorded automatically: ${tracker.total_cost:.6f} "
              f"({tracker.records[0].input_tokens} in / "
              f"{tracker.records[0].output_tokens} out)")

    print()
    print("=" * 68)
    print("4. The budget guard stops a call before the money is spent")
    print("=" * 68)
    calls_made = []

    @track_cost(model="claude-opus-4-1", tag="runaway")
    def expensive(messages: str) -> object:
        calls_made.append(1)  # never reached
        return client.messages.create(model="claude-opus-4-1", messages=[])

    with CostTracker(budget=10.00, per_request=0.05):
        try:
            expensive("a very long document " * 40_000)
        except BudgetExceededError as exc:
            print(f"  blocked on the {exc.limit_type} limit")
            print(f"    limit     ${exc.limit:.4f}")
            print(f"    projected ${exc.projected:.4f}")
            print(f"    the wrapped function ran {len(calls_made)} times, so $0 was spent")

    print()
    print("=" * 68)
    print("5. Custom pricing for a model tokentab has never heard of")
    print("=" * 68)
    register_model("acme-llm-v2", input=0.40, output=1.20, provider="acme")
    print(f"  acme-llm-v2, 1M in + 1M out: ${estimate_cost('acme-llm-v2', 10**6, 10**6):.2f}")

    print()
    print("=" * 68)
    print("6. Reporting")
    print("=" * 68)
    with CostTracker(budget=5.00, name="pipeline") as tracker:
        tracker.record("claude-sonnet-4-5",
                       TokenUsage(input_tokens=1200, output_tokens=450, cache_read_tokens=8000),
                       tag="summarize")
        tracker.record("gpt-4o-mini", TokenUsage(input_tokens=800, output_tokens=120),
                       tag="classify")
        tracker.record("gpt-4o-mini", TokenUsage(input_tokens=650, output_tokens=90),
                       tag="classify")
        tracker.record("gemini-2.5-flash", TokenUsage(input_tokens=15_000, output_tokens=800),
                       tag="extract")

    print(tracker.report().summary())
    print()
    print("  cost by tag:", {k: round(v, 6) for k, v in tracker.report().cost_by_tag().items()})
    compact = tracker.report().to_json(indent=None, include_records=False)
    print("  as JSON:    ", compact[:88], "...")
    try:
        frame = tracker.report().to_dataframe()
        print(f"  as DataFrame: {len(frame)} rows x {len(frame.columns)} columns")
    except ImportError:
        print("  as DataFrame: install the pandas extra to enable")


if __name__ == "__main__":
    main()
