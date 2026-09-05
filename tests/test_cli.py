from __future__ import annotations

import json

import pytest

from tokentab.__main__ import main


def test_price_text(capsys):
    assert main(["price", "gpt-4o", "-i", "1000000", "-o", "0"]) == 0
    out = capsys.readouterr().out
    assert "$2.500000" in out


def test_price_json(capsys):
    assert main(["price", "claude-sonnet-4-5", "-i", "1000", "-o", "500", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cost"] == pytest.approx(0.0105)
    assert data["pricing"]["provider"] == "anthropic"


def test_price_unknown_model(capsys):
    assert main(["price", "not-a-model"]) == 2
    assert "No pricing entry" in capsys.readouterr().err


def test_models_listing(capsys):
    assert main(["models", "claude"]) == 0
    assert "claude-sonnet-4-5" in capsys.readouterr().out
    assert main(["models", "zzzz"]) == 1


def test_meta(capsys):
    assert main(["meta"]) == 0
    assert json.loads(capsys.readouterr().out)["currency"] == "USD"


def test_custom_pricing_file(tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"models": {"gpt-4o": {"input": 100.0, "output": 100.0}}}))
    assert main(["--pricing-file", str(path), "price", "gpt-4o", "-i", "1000000", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cost"] == pytest.approx(100.0)
