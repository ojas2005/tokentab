from __future__ import annotations

import json

import pytest

from tokentab import CostTracker, TokenUsage


@pytest.fixture
def tracker():
    tracker = CostTracker(budget=10.0, name="session")
    tracker.record("claude-sonnet-4-5", TokenUsage(input_tokens=1000, output_tokens=500,
                                                   cache_read_tokens=4000), tag="rag")
    tracker.record("gpt-4o-mini", TokenUsage(input_tokens=2000, output_tokens=100), tag="rag")
    tracker.record("gpt-4o-mini", TokenUsage(input_tokens=500, output_tokens=50), tag="classify")
    return tracker


def test_totals(tracker):
    report = tracker.report()
    assert report.call_count == 3
    assert report.input_tokens == 3500
    assert report.output_tokens == 650
    assert report.cache_read_tokens == 4000
    assert report.total_cost == pytest.approx(tracker.total_cost)


def test_group_by_model(tracker):
    by_model = tracker.report().by_model
    assert set(by_model) == {"claude-sonnet-4-5", "gpt-4o-mini"}
    assert by_model["gpt-4o-mini"].calls == 2
    assert by_model["gpt-4o-mini"].input_tokens == 2500
    # Groups are ordered most expensive first.
    assert list(by_model) == ["claude-sonnet-4-5", "gpt-4o-mini"]


def test_group_by_tag_and_provider(tracker):
    report = tracker.report()
    assert report.by_tag["rag"].calls == 2
    assert report.by_tag["classify"].calls == 1
    assert set(report.by_provider) == {"anthropic", "openai"}
    assert sum(report.cost_by_tag().values()) == pytest.approx(report.total_cost)


def test_filter(tracker):
    narrowed = tracker.report().filter(tag="classify")
    assert narrowed.call_count == 1
    assert narrowed.total_cost < tracker.total_cost


def test_to_dict(tracker):
    data = tracker.report().to_dict()
    assert data["call_count"] == 3
    assert data["name"] == "session"
    assert data["limits"]["per_session"] == 10.0
    assert len(data["records"]) == 3
    assert "started_at" in data and "ended_at" in data
    assert data["by_model"]["gpt-4o-mini"]["calls"] == 2

    slim = tracker.report().to_dict(include_records=False)
    assert "records" not in slim


def test_to_json_round_trips(tracker):
    parsed = json.loads(tracker.report().to_json())
    assert parsed["call_count"] == 3
    assert parsed["records"][0]["model"] == "claude-sonnet-4-5"
    assert isinstance(parsed["records"][0]["timestamp"], str)


def test_records_are_flat_and_complete(tracker):
    row = tracker.report().to_records()[0]
    for key in ("request_id", "timestamp", "model", "provider", "tag", "estimated",
                "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                "input_cost", "output_cost", "cost", "total_tokens"):
        assert key in row, key


def test_summary_text(tracker):
    text = tracker.report().summary()
    assert "tokentab report [session]" in text
    assert "calls           3" in text
    assert "by model" in text and "by tag" in text
    assert "session budget" in text


def test_empty_report():
    report = CostTracker().report()
    assert report.call_count == 0
    assert report.total_cost == 0.0
    assert report.by_model == {}
    assert "calls           0" in report.summary()
    assert json.loads(report.to_json())["records"] == []


def test_estimated_share(tracker):
    tracker.record("gpt-4o", TokenUsage(input_tokens=1_000_000), estimated=True)
    assert 0 < tracker.report().estimated_cost_share < 1


def test_to_dataframe():
    pytest.importorskip("pandas")
    tracker = CostTracker()
    tracker.record("gpt-4o", TokenUsage(input_tokens=1000, output_tokens=100), tag="a")
    tracker.record("gpt-4o", TokenUsage(input_tokens=2000, output_tokens=200), tag="b")
    frame = tracker.report().to_dataframe()
    assert len(frame) == 2
    assert frame["cost"].sum() == pytest.approx(tracker.total_cost)
    assert str(frame["timestamp"].dtype).startswith("datetime64")
    assert set(frame["tag"]) == {"a", "b"}
    grouped = frame.groupby("model")["cost"].sum()
    assert grouped["gpt-4o"] == pytest.approx(tracker.total_cost)


def test_empty_dataframe_has_columns():
    pytest.importorskip("pandas")
    frame = CostTracker().report().to_dataframe()
    assert frame.empty
    assert "cost" in frame.columns
