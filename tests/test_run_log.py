"""Tests for the run log and the cost calculation."""
import datetime as dt
import json

import pytest

from agent.contract import RunResult, ToolCall
from agent.run_log import cost_usd, log_run


def make_result(question="q"):
    return RunResult(question=question, answer="a", model="claude-haiku-5-5",
                     prompt_version="v1",
                     tool_calls=[ToolCall("get_accuracy", {}, {"backtest_weeks": 8}, 0.01)],
                     input_tokens=100, output_tokens=20, seconds=1.5, stop_reason="answered")


def test_log_run_writes_one_json_line_per_run(tmp_path):
    path = tmp_path / "nested" / "runs.jsonl"  # folder does not exist yet
    log_run(make_result("first"), path=path)
    log_run(make_result("Wordt het druk?"), path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [r["question"] for r in records] == ["first", "Wordt het druk?"]
    assert records[0]["tool_calls"][0]["name"] == "get_accuracy"
    assert records[0]["stop_reason"] == "answered"
    stamp = dt.datetime.fromisoformat(records[0]["timestamp"])
    assert stamp.utcoffset() == dt.timedelta(0)


def test_cost_haiku():
    # 1M in * 0.10 + 1M out * 0.50
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("claude-haiku-5-5", 2000, 300) == pytest.approx(0.00035)


def test_cost_sonnet():
    assert cost_usd("claude-sonnet-5-5", 2000, 300) == pytest.approx(0.007)


def test_cost_unknown_model_is_none():
    assert cost_usd("some-other-model", 100, 100) is None
