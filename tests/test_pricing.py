from __future__ import annotations

import json

import pytest

from tokentab import (
    ModelPricing,
    PricingError,
    PricingRegistry,
    TokenUsage,
    UnknownModelError,
    detect_provider,
    estimate_cost,
    normalize_model,
    register_model,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("anthropic/claude-opus-4-1", "claude-opus-4-1"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("GPT-4O-MINI", "gpt-4o-mini"),
        ("claude-3-5-haiku-latest", "claude-3-5-haiku"),
    ],
)
def test_normalize_model(raw, expected):
    assert normalize_model(raw) == expected


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-sonnet-4-5", "anthropic"),
        ("gpt-4o", "openai"),
        ("o3-mini", "openai"),
        ("gemini-2.5-pro", "google"),
        ("llama-3-70b", "unknown"),
    ],
)
def test_detect_provider(model, provider):
    assert detect_provider(model) == provider


def test_basic_cost_math(registry):
    # 1M in + 1M out at sonnet rates = $3 + $15.
    assert registry.estimate("claude-sonnet-4-5", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert registry.estimate("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)


def test_cache_tokens_priced_separately(registry):
    """Anthropic cache reads are 10% of input; cache writes are 125%."""
    usage = TokenUsage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    assert registry.cost("claude-sonnet-4-5", usage) == pytest.approx(0.30)

    usage = TokenUsage(cache_write_tokens=1_000_000)
    assert registry.cost("claude-sonnet-4-5", usage) == pytest.approx(3.75)

    breakdown = registry.breakdown(
        "claude-sonnet-4-5",
        TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000,
                   cache_read_tokens=1_000_000, cache_write_tokens=1_000_000),
    )
    assert breakdown.input_cost == pytest.approx(3.0)
    assert breakdown.output_cost == pytest.approx(15.0)
    assert breakdown.cache_read_cost == pytest.approx(0.30)
    assert breakdown.cache_write_cost == pytest.approx(3.75)
    assert breakdown.total == pytest.approx(22.05)


def test_cache_rates_fall_back_to_input_rate():
    pricing = ModelPricing(input=10.0, output=20.0)
    assert pricing.effective_cache_read == 10.0
    assert pricing.effective_cache_write == 10.0
    assert pricing.cost(TokenUsage(cache_read_tokens=1_000_000)).total == pytest.approx(10.0)


def test_unknown_model_raises_actionable_error(registry):
    with pytest.raises(UnknownModelError) as info:
        registry["some-local-model"]
    assert "register_model" in str(info.value)
    assert registry.get("some-local-model") is None


def test_fallback_pricing():
    registry = PricingRegistry(fallback=ModelPricing(input=1.0, output=2.0))
    assert registry.estimate("anything-at-all", 1_000_000, 0) == pytest.approx(1.0)


def test_register_and_override(registry):
    registry.register("my-finetune", input=0.5, output=1.5)
    assert registry.estimate("my-finetune", 1_000_000, 1_000_000) == pytest.approx(2.0)
    # Overriding a bundled model works too.
    registry.register("gpt-4o", input=1.0, output=1.0)
    assert registry.estimate("gpt-4o", 1_000_000, 0) == pytest.approx(1.0)


def test_register_requires_prices(registry):
    with pytest.raises(PricingError):
        registry.register("bad-model")


def test_load_file_overrides(tmp_path, registry):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({
        "models": {"claude-sonnet-4-5": {"input": 1.0, "output": 2.0, "cache_read": 0.1}},
        "aliases": {"my-alias": "claude-sonnet-4-5"},
    }))
    registry.load_file(path)
    assert registry.estimate("claude-sonnet-4-5", 1_000_000, 0) == pytest.approx(1.0)
    assert registry.estimate("my-alias", 0, 1_000_000) == pytest.approx(2.0)


def test_load_file_errors(tmp_path, registry):
    missing = tmp_path / "nope.json"
    with pytest.raises(PricingError, match="not found"):
        registry.load_file(missing)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(PricingError, match="valid JSON"):
        registry.load_file(bad)
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"models": {"x": {"input": 1.0}}}))
    with pytest.raises(PricingError, match="missing required key"):
        registry.load_file(incomplete)


def test_negative_price_rejected():
    with pytest.raises(PricingError):
        ModelPricing(input=-1.0, output=1.0)


def test_env_var_override(tmp_path, monkeypatch):
    path = tmp_path / "env.json"
    path.write_text(json.dumps({"models": {"gpt-4o": {"input": 99.0, "output": 99.0}}}))
    monkeypatch.setenv("TOKENTAB_PRICING_FILE", str(path))
    registry = PricingRegistry()
    assert registry.estimate("gpt-4o", 1_000_000, 0) == pytest.approx(99.0)


def test_copy_is_independent(registry):
    clone = registry.copy()
    clone.register("gpt-4o", input=0.0, output=0.0)
    assert registry.estimate("gpt-4o", 1_000_000, 0) == pytest.approx(2.5)
    assert clone.estimate("gpt-4o", 1_000_000, 0) == pytest.approx(0.0)


def test_unseen_dated_snapshot_resolves_by_prefix(registry):
    """A future dated release of a known model should still price."""
    assert registry.estimate("claude-sonnet-4-5-20991231", 1_000_000, 0) == pytest.approx(3.0)


def test_module_level_helpers():
    register_model("test-only-model", input=2.0, output=4.0)
    assert estimate_cost("test-only-model", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_bundled_data_is_sane(registry):
    assert len(registry) >= 30
    assert registry.meta["currency"] == "USD"
    for name, pricing in registry.models().items():
        assert pricing.input >= 0 and pricing.output >= 0, name
        # Output is never cheaper than input for chat models.
        if not name.startswith("text-embedding"):
            assert pricing.output >= pricing.input, name
