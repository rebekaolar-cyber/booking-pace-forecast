"""End-to-end test of evals/run.py with a stub `answer` (no API, no key needed)."""
from __future__ import annotations

import json
import sys
import types

import pytest

from agent.contract import RunResult, ToolCall, run_tool
from evals import compare, run


def oracle_answer(question, backend, client=None, model="m", prompt_version="v1", **kw):
    """A perfect assistant: makes the reference calls and states the expected numbers."""
    item = next(q for q in run.load_questions() if q["question"] == question)
    expect = item["expect"]
    result = RunResult(question=question, answer="", model=model, prompt_version=prompt_version,
                       input_tokens=1000, output_tokens=100, seconds=0.5, stop_reason="answered")
    if any(t == "get_today_and_data_range" for t in expect.get("tools") or []):
        out = run_tool(backend, "get_today_and_data_range", {})
        result.tool_calls.append(ToolCall("get_today_and_data_range", {}, out, 0.0))
    numbers = []
    for call in expect.get("calls") or []:
        out = run_tool(backend, call["tool"], call.get("args") or {})
        result.tool_calls.append(ToolCall(call["tool"], call.get("args") or {}, out, 0.0))
        for field in expect.get("numbers_from") or []:
            numbers += [str(v) for v in run.scoring._values_for(out, field)]
    if expect.get("must_refuse"):
        result.answer = ("Helaas, dat kan ik niet voorspellen." if item["lang"] == "nl"
                         else "Sorry, I can't answer that; I can only forecast guests and staff.")
    elif item["lang"] == "nl":
        result.answer = f"We verwachten voor die dag: {', '.join(numbers) or 'geen bijzonderheden'}. Gebaseerd op de tools."
    else:
        result.answer = f"We expect the following for the days you asked: {', '.join(numbers) or 'nothing unusual'}."
    return result


def liar_answer(question, backend, **kw):
    r = oracle_answer(question, backend, **kw)
    r.answer += " Also expect 9999 guests at the bar."
    return r


@pytest.fixture
def stub_agent(monkeypatch):
    """Install stub agent.loop / agent.run_log modules so the runner's lazy imports find them."""
    def install(answer_fn):
        loop = types.ModuleType("agent.loop")
        loop.answer = answer_fn
        log = types.ModuleType("agent.run_log")
        log.cost_usd = lambda model, i, o: (i * 0.10 + o * 0.50) / 1_000_000
        monkeypatch.setitem(sys.modules, "agent.loop", loop)
        monkeypatch.setitem(sys.modules, "agent.run_log", log)
    return install


def test_full_run_with_perfect_stub(stub_agent, tmp_path, capsys):
    stub_agent(oracle_answer)
    code = run.main(["--backend", "fake", "--out", str(tmp_path)])
    assert code == 0
    files = list(tmp_path.glob("*_claude-haiku-5-5_v1.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    s = data["summary"]
    assert s["questions"] == 50
    assert s["pass_rate"] == 1.0, [r["id"] for r in data["results"] if not r["passed"]]
    assert s["metrics"]["refusal"]["applicable"] == 8  # 7 out_of_scope + 1 Dutch
    assert s["total_cost_usd"] == pytest.approx(50 * 0.00015)
    assert s["cost_per_100_questions_usd"] == pytest.approx(0.015)
    assert s["p50_seconds"] == 0.5
    loose = {r["id"] for r in data["results"] if r["faithfulness_loose"]}
    assert s["faithfulness_loose"] == len(loose)  # multi-day answers that name no day
    out = capsys.readouterr().out
    assert "faithfulness" in out and "per 100 questions" in out


def test_invented_number_fails_the_run(stub_agent, tmp_path):
    stub_agent(liar_answer)
    code = run.main(["--ids", "sd01", "st01", "--out", str(tmp_path)])
    assert code == 1
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["summary"]["metrics"]["faithfulness"]["rate"] == 0.0
    assert all(r["ungrounded"] == [9999.0] for r in data["results"])
    assert all(r["ungrounded_detail"][0]["number"] == 9999.0 for r in data["results"])


def test_sample_is_stratified_and_fixed():
    questions = run.load_questions()
    a = run.stratified_sample(questions, 10, seed=0)
    assert [q["id"] for q in a] == [q["id"] for q in run.stratified_sample(questions, 10, seed=0)]
    assert len(a) == 10
    assert {q["category"] for q in a} == {q["category"] for q in questions}  # every category
    assert [q["id"] for q in a] == [q["id"] for q in questions if q in a]  # file order
    assert run.stratified_sample(questions, 10, seed=1) != a
    assert len(run.stratified_sample(questions, 3)) == 3
    assert run.stratified_sample(questions, 99) == questions


def test_sample_option(stub_agent, tmp_path):
    stub_agent(oracle_answer)
    assert run.main(["--sample", "10", "--out", str(tmp_path)]) == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["config"]["sample"] == 10 and data["config"]["seed"] == 0
    expected = [q["id"] for q in run.stratified_sample(run.load_questions(), 10, 0)]
    assert [r["id"] for r in data["results"]] == expected


def test_limit_and_ids(stub_agent, tmp_path):
    stub_agent(oracle_answer)
    assert run.main(["--limit", "3", "--out", str(tmp_path)]) == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert [r["id"] for r in data["results"]] == ["sd01", "sd03", "sd04"]
    with pytest.raises(SystemExit):
        run.main(["--ids", "nope", "--out", str(tmp_path)])


def test_real_backend_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.real_backend", None)  # import fails
    with pytest.raises(SystemExit, match="agent/real_backend.py"):
        run.make_backend("real")


def test_api_error_fails_the_run(stub_agent, tmp_path):
    def broken(question, backend, model="m", prompt_version="v1", **kw):
        return RunResult(question=question, answer="Sorry, I couldn't reach the AI service.",
                         model=model, prompt_version=prompt_version, stop_reason="error",
                         error="no API key")
    stub_agent(broken)
    assert run.main(["--ids", "sd01", "--out", str(tmp_path)]) == 1


def test_percentile():
    assert run.percentile([1, 2, 3, 4, 100], 50) == 3
    assert run.percentile([1, 2, 3, 4, 100], 95) == 100
    assert run.percentile([], 50) is None


def test_compare_table(stub_agent, tmp_path):
    stub_agent(oracle_answer)
    run.main(["--limit", "2", "--out", str(tmp_path / "a")])
    run.main(["--limit", "2", "--prompt", "v2", "--out", str(tmp_path / "b")])
    paths = [next((tmp_path / "a").glob("*.json")), next((tmp_path / "b").glob("*.json"))]
    table = compare.compare_table(paths)
    lines = table.splitlines()
    assert len(lines) == 4 and lines[0].startswith("| Run")
    assert "/ v2 /" in lines[3]


def test_alt_reading_is_scored_against_the_date_the_assistant_used():
    """'Next Saturday' said on a Thursday: 06-03 and 06-10 are both fair readings."""
    from agent.contract import RunResult, ToolCall
    from evals.run import pick_reading
    item = {"id": "x", "expect": {
        "calls": [{"tool": "get_staffing", "args": {"date": "2017-06-03", "role": "kitchen"}}],
        "alt_calls": [{"tool": "get_staffing", "args": {"date": "2017-06-10", "role": "kitchen"}}]}}

    def ran(date):
        return RunResult(question="q", answer="a", model="m", prompt_version="v1",
                         tool_calls=[ToolCall("get_staffing", {"date": date, "role": "kitchen"}, {}, 0.1)])

    assert pick_reading(item, ran("2017-06-03"))["expect"]["calls"][0]["args"]["date"] == "2017-06-03"
    assert pick_reading(item, ran("2017-06-10"))["expect"]["calls"][0]["args"]["date"] == "2017-06-10"
    assert pick_reading(item, ran("2017-06-17"))["expect"]["calls"][0]["args"]["date"] == "2017-06-03"
