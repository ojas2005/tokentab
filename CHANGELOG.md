# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-05

Initial release.

### Added
- **Pricing registry** for Anthropic, OpenAI and Google Gemini models, loaded
  from a bundled JSON file and overridable per model, per file, per registry, or
  via the `TOKENTAB_PRICING_FILE` environment variable.
- **Separate cache-read and cache-write rates**, so Anthropic prompt caching is
  priced correctly rather than at the full input rate.
- **Model id normalization**, resolving gateway and cloud ids
  (`us.anthropic.claude-...-v1:0`, `openai/gpt-4o`, `models/gemini-2.5-flash`)
  and dated snapshots to their price entry.
- **Pre-call token estimation** with `tiktoken` for OpenAI, Anthropic's
  `count_tokens` endpoint for Claude, and a dependency-free character heuristic
  everywhere else.
- **Post-call usage extraction** from Anthropic, OpenAI, Gemini and LangChain
  response shapes, normalizing the two conflicting cached-token conventions.
- **Budget guard** with per-request, per-session and per-day limits, warn-only
  mode, and thread-safe counters including an atomic `reserve()`.
- **Three integration surfaces** sharing one ledger: `@track_cost`,
  `CostTracker` as a context manager, and `TokenTabCallbackHandler` for
  LangChain.
- **Reporting** with per-request breakdowns, aggregates by model, tag and
  provider, and export to dict, JSON and pandas DataFrame.
- **CLI**: `tokentab price`, `tokentab models`, `tokentab meta`.

[Unreleased]: https://github.com/ojas2005/tokentab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ojas2005/tokentab/releases/tag/v0.1.0
