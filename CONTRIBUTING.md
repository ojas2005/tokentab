# Contributing

Thanks for helping out. This project is small and intends to stay that way.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,all]'
```

## Before opening a PR

```bash
pytest          # the full suite, extras included
mypy            # strict, must be clean
ruff check .
```

## Ground rules

- **The core stays dependency-free.** `tiktoken`, `anthropic`, `pandas` and
  `langchain-core` are optional extras. Import them lazily, inside the function
  or module that needs them, and always provide a working fallback or a clear
  `ImportError` naming the extra to install.
- **Never swallow a provider exception.** Anything wrapping a user's API call
  must let its exceptions through untouched.
- **Never let bookkeeping break a paid-for call.** After a successful request,
  a tokentab-internal error is downgraded to a warning.
- **Python 3.9 is the floor.** Use `from __future__ import annotations` plus
  `typing.Optional`/`Dict`/`List`, not PEP 604 unions in runtime positions.
- New behavior needs a test. Concurrency-sensitive changes need a test that
  actually spawns threads.

## Updating prices

Edit `src/tokentab/data/pricing.json`, in USD per million tokens, and bump
`_meta.as_of`. Include a link to the provider's public pricing page in the PR.
Prices are best-effort public list prices; users who need exactness override
them at runtime.
