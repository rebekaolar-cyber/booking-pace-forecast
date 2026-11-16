"""Checks on evals/questions.yaml: shape, counts, and that every reference call works."""
from __future__ import annotations

import datetime as dt
from collections import Counter

import pytest

from agent.contract import TOOL_NAMES, run_tool
from agent.fake_backend import FakeBackend
from evals.run import load_questions

QUESTIONS = load_questions()
IN_SCOPE = [q for q in QUESTIONS if not q["expect"].get("must_refuse")]
FIRST, LAST = dt.date(2017, 6, 1), dt.date(2017, 6, 28)

EXPECTED_COUNTS = {"single_day": 8, "date_range": 8, "staffing": 8, "dutch": 8,
                   "out_of_scope": 7, "relative_dates": 8, "trust_accuracy": 3}
NL_OUTSIDE_DUTCH = {"rd08"}  # "volgende zaterdag": a relative date asked in Dutch


def _any_of(expect: dict) -> set[str]:
    return {t for option in expect.get("tools_any_of") or [] for t in option}


def test_fifty_questions_with_unique_ids():
    assert len(QUESTIONS) == 50
    ids = [q["id"] for q in QUESTIONS]
    assert len(set(ids)) == len(ids)


def test_category_counts():
    assert Counter(q["category"] for q in QUESTIONS) == EXPECTED_COUNTS


@pytest.mark.parametrize("item", QUESTIONS, ids=lambda q: q["id"])
def test_item_shape(item):
    assert set(item) >= {"id", "category", "question", "lang", "expect"}
    assert item["lang"] in ("en", "nl")
    expect = item["expect"]
    for tool in set(expect.get("tools") or []) | set(expect.get("allowed_tools") or []) | _any_of(expect):
        assert tool in TOOL_NAMES
    for option in expect.get("tools_any_of") or []:
        assert isinstance(option, list) and option
    for call in expect.get("calls") or []:
        assert call["tool"] in TOOL_NAMES
    if expect.get("must_refuse"):
        assert not expect.get("tools") and not expect.get("calls")
    else:
        assert (expect.get("tools") or expect.get("tools_any_of")) and expect.get("calls")
        called = {c["tool"] for c in expect["calls"]}
        assert called <= set(expect.get("tools") or []) | _any_of(expect) | {"get_today_and_data_range"}


def test_dutch_category_is_in_dutch():
    assert all(q["lang"] == "nl" for q in QUESTIONS if q["category"] == "dutch")
    others_nl = {q["id"] for q in QUESTIONS if q["category"] != "dutch" and q["lang"] == "nl"}
    assert others_nl == NL_OUTSIDE_DUTCH


def test_flag_questions_accept_get_forecast():
    """get_forecast returns the flag of every day, so it must be an accepted way to answer."""
    for q in QUESTIONS:
        if any(c["tool"] == "get_flagged_days" for c in q["expect"].get("calls") or []):
            assert [["get_flagged_days"], ["get_forecast"]] == q["expect"].get("tools_any_of"), q["id"]
            assert "get_flagged_days" not in (q["expect"].get("tools") or []), q["id"]


def test_next_saturday_items_read_as_2017_06_03():
    items = [q for q in QUESTIONS if "next saturday" in q["question"].lower()
             or "volgende zaterdag" in q["question"].lower()]
    assert {q["lang"] for q in items} == {"en", "nl"}
    for q in items:
        assert "2017-06-03" in q["expect"]["note"]
        dates = {v for c in q["expect"]["calls"] for k, v in c["args"].items() if "date" in k}
        assert dates == {"2017-06-03"}
        assert dt.date(2017, 6, 1).weekday() == 3 and dt.date(2017, 6, 3).weekday() == 5


def test_out_of_scope_items_must_refuse():
    for q in QUESTIONS:
        if q["category"] == "out_of_scope":
            assert q["expect"].get("must_refuse") is True, q["id"]


def test_relative_items_have_a_note_and_need_today():
    for q in QUESTIONS:
        if q["category"] == "relative_dates":
            assert q["expect"].get("note"), q["id"]
            assert "get_today_and_data_range" in q["expect"]["tools"], q["id"]


@pytest.mark.parametrize("item", IN_SCOPE, ids=lambda q: q["id"])
def test_reference_calls_work_on_fake_backend(item):
    backend = FakeBackend()
    for call in item["expect"]["calls"]:
        args = call.get("args") or {}
        for key in ("date", "start_date", "end_date"):
            if key in args:
                assert FIRST <= dt.date.fromisoformat(args[key]) <= LAST
        out = run_tool(backend, call["tool"], args)
        assert "error" not in out, (item["id"], out)
        for field in item["expect"].get("numbers_from") or []:
            assert field in repr(out) or any(
                field in repr(run_tool(backend, c["tool"], c.get("args") or {}))
                for c in item["expect"]["calls"]), (item["id"], field)


def test_weekday_names_in_questions_match_dates():
    """'Saturday 3 June 2017' must really be a Saturday."""
    import re
    months = {"june": 6, "juni": 6}
    days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5,
            "sunday": 6, "maandag": 0, "dinsdag": 1, "woensdag": 2, "donderdag": 3, "vrijdag": 4,
            "zaterdag": 5, "zondag": 6}
    pattern = re.compile(r"(" + "|".join(days) + r")\s+(\d{1,2})\s+(june|juni)\s+2017", re.I)
    checked = 0
    for q in QUESTIONS:
        for wd, day, month in pattern.findall(q["question"]):
            assert dt.date(2017, months[month.lower()], int(day)).weekday() == days[wd.lower()], q["id"]
            checked += 1
    assert checked >= 10
