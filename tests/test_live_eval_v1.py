"""The 22 failures of the first full live eval (prompt v1, 50 questions), as regression tests.

tests/data/live_v1_failures.json holds each failed question with the model's exact answer and
the tool calls it made. The tool outputs are rebuilt here with RealBackend (today 2017-06-01),
and each case is scored with the same code as evals/run.py.

Triage (also in the docstrings of the scoring rules they led to):
* 16 were scorer false alarms or unfair checks; they must now pass.
* 6 are real model mistakes and must keep failing, on exactly the listed checks:
  sd05 "about 10%" for +10.5% (half-up gives 11), sd09 called get_accuracy for a plain forecast,
  dr07 "three quiet days" (one is flagged quiet) and "about 15% below" for -14.5% (unflagged),
  nl03 a garbled word ("минимaal") where the threshold is named, so 15% reads as a day's change,
  st06 and rd04 answered an English question in Dutch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pace.plan")

from agent.contract import RunResult, ToolCall, run_tool  # noqa: E402
from agent.real_backend import RealBackend  # noqa: E402
from evals import run  # noqa: E402

CASES = json.loads((Path(__file__).parent / "data" / "live_v1_failures.json").read_text(encoding="utf-8"))
ITEMS = {q["id"]: q for q in run.load_questions()}


@pytest.fixture(scope="module")
def real():
    return RealBackend()


def test_all_22_failures_are_triaged():
    assert len(CASES) == 22
    assert sum(1 for c in CASES if c["still_fails"]) == 6


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_live_v1_failure(case, real):
    result = RunResult(question=case["question"], answer=case["answer"], model="live",
                       prompt_version="v1", stop_reason="answered",
                       tool_calls=[ToolCall(c["tool"], c["args"], run_tool(real, c["tool"], c["args"]), 0.0)
                                   for c in case["calls"]])
    item = run.pick_reading(ITEMS[case["id"]], result)
    reference = [run_tool(real, c["tool"], c.get("args") or {}) for c in item["expect"].get("calls") or []]
    scored = run.score_question(item, result, reference)
    failed = sorted(k for k, v in scored["scores"].items() if v is False)
    assert failed == sorted(case["still_fails"]), scored["ungrounded_detail"]
