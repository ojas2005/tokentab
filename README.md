# tokentab

**Know what an LLM call will cost before you make it — and stop the ones you can't afford.**

`tokentab` wraps LLM API calls to estimate cost up front, enforce budgets, and
record what was actually spent. It has **zero required dependencies**, works with
Anthropic, OpenAI and Google Gemini out of the box, and never needs an API key of
its own.

```python
from tokentab import CostTracker

with CostTracker(budget=5.00) as tracker:
    response = client.messages.create(model="claude-sonnet-4-5", messages=messages)
    tracker.record_response(response, "claude-sonnet-4-5")

print(tracker.report().summary())
```

```
tokentab report
=================
calls           1
total cost      $0.010500
total tokens    1,500  (in 1,000 / out 500 / cache r 0 w 0)
session budget  $0.010500 / $5.000000 (0.2%)
```

---

## Why

Most cost trackers tell you what you spent *after* you spent it. `tokentab`
counts the prompt **before** the request goes out, prices it, and raises
`BudgetExceededError` if it would breach your limit — so a runaway agent loop
costs you nothing instead of your monthly budget.

- **Pre-call enforcement.** The guard blocks before the money is spent, not after.
- **Real numbers after the fact.** Post-call, the provider's own usage object is
  read for the true cost, including cache-read and cache-write tokens at their
  own rates.
- **Three ways in, one ledger.** Decorator, context manager, and LangChain
  callback handler all share the same tracker and budget.
- **Stays out of the way.** Exceptions from your provider SDK pass through
  untouched, and a bookkeeping failure never destroys a response you paid for.

## Install

```bash
pip install tokentab
```

The core is pure standard library. Extras are opt-in:

```bash
pip install 'tokentab[tiktoken]'   # exact token counts for OpenAI models
pip install 'tokentab[anthropic]'  # exact counts via Anthropic's count_tokens endpoint
pip install 'tokentab[pandas]'     # Report.to_dataframe()
pip install 'tokentab[langchain]'  # LangChain callback handler
pip install 'tokentab[all]'        # everything
```

Python 3.9+.

---

## The three integration surfaces

### 1. Context manager

```python
from tokentab import CostTracker

with CostTracker(budget=5.00, per_request=0.50, per_day=25.00) as tracker:
    response = client.messages.create(...)
    tracker.record_response(response, "claude-sonnet-4-5", tag="summarize")

    print(tracker.total_cost)          # 0.0105
    print(tracker.status().session_remaining)
```

To enforce the budget *before* the call, wrap it in `tracker.call(...)`:

```python
with CostTracker(budget=5.00) as tracker:
    # Raises BudgetExceededError here if the prompt is too expensive to send.
    with tracker.call("claude-sonnet-4-5", messages=messages, expected_output_tokens=1000) as call:
        call.set_response(client.messages.create(model="claude-sonnet-4-5", messages=messages))
```

`expected_output_tokens` is your allowance for the response, and it matters more
than it looks: output costs 3–5× input, so a guard that prices only the prompt
will let a session budget overshoot by roughly one full response. Set it to your
`max_tokens` for a hard ceiling, or to a realistic average for a looser one.

### 2. Decorator

```python
from tokentab import track_cost

@track_cost(model="gpt-4o", tag="summarize")
def summarize(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)
```

The decorator finds the prompt in the parameter named `messages`, `prompt`,
`input`, `text`, `contents` or `query` (or one you name with `messages_arg=`),
estimates it, enforces the active budget, then reads the real usage off the
return value. It works on `async def` too, and reads the model from the
function's own arguments when you don't pass one:

```python
@track_cost(tag="research")          # model comes from the `model` argument
async def call(model, messages): ...
```

### 3. LangChain callback handler

```python
from tokentab import TokenTabCallbackHandler

handler = TokenTabCallbackHandler(budget=5.00, tag="research-agent")
agent.invoke({"input": "..."}, config={"callbacks": [handler]})

print(handler.report().summary())
```

Every model call the chain or agent makes is priced on `on_llm_start` (raising
before the provider is called if it would breach the budget) and recorded with
real usage on `on_llm_end`. The pre-call estimate budgets for the response too,
using your `expected_output_tokens=` if you pass one and the model's configured
`max_tokens` otherwise.

### Sharing one budget across all three

Pass the same tracker, or just nest them — an active `CostTracker` is picked up
automatically by decorated functions and by handlers created without their own
budget:

```python
with CostTracker(budget=10.00, per_day=100.00) as run:
    summarize(messages)                                   # decorated function
    agent.invoke(payload, config={"callbacks": [TokenTabCallbackHandler()]})
    run.record_response(raw_response, "gemini-2.5-flash") # a hand-rolled call

    print(run.report().cost_by_tag())
```

Trackers nest, and an inner tracker reports its spend to the enclosing one, so a
per-task budget and a per-run budget can both apply:

```python
with CostTracker(budget=100.00, name="run") as run:
    for task in tasks:
        with CostTracker(budget=1.00, name=task.id):   # ...and each task capped at $1
            process(task)
```

---

## Budgets

```python
CostTracker(
    budget=5.00,        # per session (this tracker's lifetime)
    per_request=0.50,   # any single call
    per_day=25.00,      # rolling UTC calendar day
    warn_only=False,    # True: warn instead of raise, and keep going
)
```

`BudgetExceededError` carries everything you need to report or retry:

```python
try:
    expensive_call(messages)
except BudgetExceededError as exc:
    exc.limit_type   # "per_request" | "per_session" | "per_day"
    exc.limit        # 0.50
    exc.current      # 0.42   spend so far
    exc.projected    # 0.67   what it would have become
    exc.estimated    # True: blocked before the call, no money spent
    exc.model, exc.tag
```

Warn-only mode swaps the exception for a `BudgetWarning` and an optional callback,
which is the right setting for a first rollout where you want the numbers but not
the outages:

```python
CostTracker(budget=5.00, warn_only=True, on_warning=lambda err: alert(err))
```

### Concurrency

Every counter is guarded by a lock, so an agent framework fanning calls out to a
thread pool still counts each one exactly once. A `with CostTracker(...)` block
covers worker threads too, not just the thread that opened it.

`check()` and the commit that follows it are individually atomic. If several
workers must not all pass the same check against the same remaining headroom, use
the guard's atomic reserve-then-reconcile instead:

```python
tracker.guard.reserve(estimated_cost)          # check + commit, atomically
...
tracker.guard.adjust(actual_cost - estimated_cost)   # reconcile after the call
```

---

## Cost calculation

Prices live in a bundled JSON file, in USD per million tokens, with cache rates
tracked separately because providers bill them differently — Anthropic charges
**1.25×** input for a cache write and **0.1×** for a cache read:

```json
{
  "models": {
    "claude-sonnet-4-5": {
      "provider": "anthropic",
      "input": 3.0, "output": 15.0,
      "cache_read": 0.30, "cache_write": 3.75
    }
  }
}
```

```python
from tokentab import estimate_cost, TokenUsage, default_registry

estimate_cost("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
# 0.0105

breakdown = default_registry().breakdown(
    "claude-sonnet-4-5",
    TokenUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=40_000),
)
breakdown.input_cost, breakdown.cache_read_cost, breakdown.total
# 0.003, 0.012, 0.0225
```

Model ids are normalized before lookup, so gateway and cloud-provider ids resolve
without extra configuration — `us.anthropic.claude-sonnet-4-5-20250929-v1:0`,
`openai/gpt-4o`, `models/gemini-2.5-flash` and `gpt-4o-2024-08-06` all find their
price. A dated snapshot of a known model that isn't in the table yet resolves by
longest prefix rather than failing.

### Overriding prices

Prices change, and your account's rates may not be list price. Override them
three ways:

```python
import tokentab

# 1. One model at a time
tokentab.register_model("gpt-4o", input=2.00, output=8.00)     # your negotiated rate
tokentab.register_model("my-finetune", input=0.50, output=1.50)

# 2. From a file or dict
tokentab.load_pricing("company_rates.json")

# 3. An isolated registry, leaving the global one untouched
from tokentab import PricingRegistry, CostTracker
registry = PricingRegistry().load_file("company_rates.json")
tracker = CostTracker(budget=5.00, registry=registry)
```

Or set `TOKENTAB_PRICING_FILE=/path/to/rates.json` to layer a file over the
bundled data at import time — useful for pinning rates in production without a
code change.

**The bundled prices are best-effort public list prices** with an `as_of` date in
`default_registry().meta`. They are a good default for engineering guardrails;
they are not a billing system. Override them if the numbers need to be exact.

Unknown models warn once and are counted as $0, so tokentab never breaks an
app over a model it hasn't heard of. Pass `strict_pricing=True` to turn that into
an `UnknownModelError`, or give the registry a `fallback=ModelPricing(...)`.

---

## Token counting

**Before the call**, tokentab estimates:

| Models | Counter | Accuracy |
| --- | --- | --- |
| OpenAI | `tiktoken`, when installed | exact |
| Claude | `AnthropicCounter`, when you wire up a client | exact |
| everything else | `HeuristicCounter` (characters ÷ 4) | typically within 10–20% |

```python
from tokentab import count_message_tokens, set_counter, AnthropicCounter

count_message_tokens(messages, "gpt-4o", system="You are terse.")

# Opt into exact Claude counts (a free but real network round trip, cached):
set_counter("anthropic", AnthropicCounter(client=anthropic.Anthropic()))
```

If that call ever fails, the counter degrades to the heuristic rather than taking
your application down with it.

**After the call**, no estimation is involved: usage comes from the provider's own
response object. `extract_usage` understands the Anthropic Messages API, the
OpenAI chat/responses APIs, Gemini, LangChain's `usage_metadata`, and plain dicts
in any of those shapes — including the difference between OpenAI's cached-token
count (inclusive of `prompt_tokens`) and Anthropic's (exclusive), so cached tokens
are never billed twice.

Records built from an estimate rather than reported usage are flagged
`estimated=True`, and `report.estimated_cost_share` tells you how much of a total
is estimate rather than fact.

---

## Reporting

```python
report = tracker.report()

report.total_cost          # 0.0472
report.call_count          # 12
report.cost_by_model()     # {'claude-sonnet-4-5': 0.041, 'gpt-4o-mini': 0.0062}
report.cost_by_tag()       # {'rag': 0.039, 'classify': 0.0082}
report.by_model["gpt-4o-mini"].input_tokens

report.to_dict()           # nested dict: totals, groupings, every record
report.to_json(indent=2)   # the same, as JSON
report.to_dataframe()      # one row per call (needs the pandas extra)
print(report.summary())    # the text block at the top of this README

report.filter(tag="rag").total_cost
```

Each record carries `model`, `provider`, `tag`, token counts split by kind, the
cost broken down by what each token was billed as, a UTC `timestamp`, the call
`duration_s`, a `request_id`, and any `metadata` you attached.

## CLI

```bash
tokentab price claude-sonnet-4-5 -i 12000 -o 800 --cache-read-tokens 40000
tokentab models gemini
tokentab meta                     # as-of date and source note for bundled prices
```

---

## Design notes

- **Zero required dependencies.** The core imports only the standard library.
  `tiktoken`, `anthropic`, `pandas` and `langchain-core` are optional extras, each
  imported lazily and each with a working fallback.
- **No API key.** tokentab never authenticates or calls a provider on its own.
  The one optional network call, Anthropic's `count_tokens`, uses a client *you*
  construct and pass in.
- **Your exceptions are yours.** Nothing wrapping a provider call ever catches,
  converts or suppresses an exception from it. Failed calls record nothing.
- **Bookkeeping is never fatal.** After a successful, already-billed call, a
  tokentab error during recording is downgraded to a warning rather than
  destroying the response.
- **Fully typed.** Ships `py.typed` and is clean under `mypy --strict`.

### Known limits

- **A budget can overshoot by up to one call.** The guard runs *before* each
  request, so it stops the call that would breach the limit — but the response
  length is not knowable in advance. If you do not supply
  `expected_output_tokens` (or a `max_tokens` the LangChain handler can read),
  only the prompt is priced, and the call that tips you over is the one whose
  output you did not budget for. Supply an output allowance to make the guard
  conservative, and treat per-session limits as a circuit breaker rather than a
  hard cap.
- Streaming responses that never report usage fall back to the pre-call estimate,
  flagged `estimated=True`. Pass the final usage chunk to
  `tracker.record(...)` when you have one.
- The heuristic counter is an estimate. For hard per-request limits on Claude,
  install the `anthropic` extra and wire up `AnthropicCounter`.
- Per-day budgets roll over on the **UTC** calendar day.
- Bundled prices cover text tokens. Image, audio, batch and long-context tier
  pricing are not modeled — register your own rates for those.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,all]'
pytest
mypy
ruff check .
```

## License

MIT
