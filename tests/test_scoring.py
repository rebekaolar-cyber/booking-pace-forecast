"""Unit tests for evals/scoring.py with hand-made RunResults (no API).

The faithfulness tables include every case from the independent tester's probes
(score_probe.py / score_probe2.py / other_probe.py), with the FakeBackend's current numbers.
FakeBackend facts used below (today = Thursday 2017-06-01):
  Sat 2017-06-03: booked 104, usual 118, forecast 113, range 104-122, vs usual -4%
  Sun 2017-06-04: forecast 82, range 72-92        Fri 2017-06-02: forecast 94, range 85-103
  Sat 2017-06-10: forecast 148, range 136-160, +25%, busy
  Staffing 2017-06-03: bar 3, kitchen 4, floor 5 (1 per 40 / 30 / 25 guests)
  Accuracy: mean abs error 8.6 guests, range coverage 80%
"""
from __future__ import annotations

import pytest

from agent.contract import RunResult, ToolCall, run_tool
from agent.fake_backend import FakeBackend
from evals import scoring

B = FakeBackend()
SAT = "2017-06-03"

TODAY = [("get_today_and_data_range", {})]
FC_SAT = [("get_forecast", {"start_date": SAT, "end_date": SAT})]
FC_FRI = [("get_forecast", {"start_date": "2017-06-02", "end_date": "2017-06-02"})]
FC_WEEKEND = [("get_forecast", {"start_date": SAT, "end_date": "2017-06-04"})]
FC_28 = [("get_forecast", {"start_date": "2017-06-01", "end_date": "2017-06-28"})]
STAFF_SAT = [("get_staffing", {"date": SAT})]
ACC = [("get_accuracy", {})]
FLAGS = [("get_flagged_days", {"start_date": "2017-06-02", "end_date": "2017-06-28"})]


def make(question: str, answer: str, calls: list[tuple[str, dict]]) -> RunResult:
    tool_calls = [ToolCall(name=n, input=a, output=run_tool(B, n, a), seconds=0.01) for n, a in calls]
    return RunResult(question=question, answer=answer, model="test", prompt_version="v1",
                     tool_calls=tool_calls, stop_reason="answered")


def sat_day() -> dict:
    return run_tool(B, "get_forecast", {"start_date": SAT, "end_date": SAT})["days"][0]


def test_fake_backend_facts_used_in_these_tables():
    d = sat_day()
    assert (d["booked_so_far_guests"], d["usual_for_weekday_guests"], d["forecast_guests"],
            d["range_low_guests"], d["range_high_guests"], d["vs_usual_pct"]) == (104, 118, 113, 104, 122, -4)
    staff = run_tool(B, "get_staffing", {"date": SAT})["staff"]
    assert [(s["role"], s["count"]) for s in staff] == [("bar", 3), ("kitchen", 4), ("floor", 5)]
    sun = run_tool(B, "get_forecast", {"start_date": "2017-06-04", "end_date": "2017-06-04"})["days"][0]
    assert (sun["forecast_guests"], sun["range_low_guests"], sun["range_high_guests"]) == (82, 72, 92)
    ten = run_tool(B, "get_forecast", {"start_date": "2017-06-10", "end_date": "2017-06-10"})["days"][0]
    assert (ten["forecast_guests"], ten["range_low_guests"], ten["range_high_guests"]) == (148, 136, 160)


# --------------------------------------------------------------------------- numbers_in

@pytest.mark.parametrize("text, expected", [
    ("Expect 118 guests (106-130).", [118, 106, 130]),
    ("On 2017-06-03 you need 3 bar staff.", [3]),
    ("Saturday 3 June 2017: 118 guests", [118]),
    ("zaterdag 3 juni: 118 gasten", [118]),
    ("June 3rd, 2017 looks normal", []),
    ("On 03/06 and 3/6 it's 80", [80]),
    ("Weekend of 17 and 18 June: 99", [99]),
    ("19 t/m 25 juni: 60", [60]),
    ("Error is 1,5 guests or 1.5 guests", [1.5, 1.5]),
    ("In 2017 there were 1,200 bookings", [2017, 1200]),  # a bare year is a number without context
    ("1. Saturday 118\n2. Sunday 80", [118, 80]),
    ("Plan:\n150. guests is the max", [150]),              # only 1-2 digit list markers
    ("Quieter by -20% at 18:00", [20]),
    ("You need three bartenders and twee koks", [3, 2]),
    ("Dit is een drukke dag", []),                         # Dutch "een" is an article
    ("June 3 - 12 extra guests", [12]),
    ("In week 23 it's busy", []),
    ("", []),
])
def test_numbers_in(text, expected):
    assert scoring.numbers_in(text) == [float(x) for x in expected]


@pytest.mark.parametrize("answer, source, ok", [
    (9, 8.6, True), (9, 8.5, True), (8, 8.4, True), (8.6, 8.6, True), (-9, -8.5, True),
    (8, 8.5, False), (10, 8.6, False), (113, 113, True), (112, 113, False),
])
def test_value_matches_rounds_half_up(answer, source, ok):
    assert scoring.value_matches(answer, source) is ok


# --------------------------------------------------------------------------- faithfulness tables

CAUGHT = [  # (label, answer, calls): must be flagged as invented
    # tester findings
    ("other day's numbers after 28-day pull", "Saturday 3 June: 63 guests (range 54-72).", FC_28),
    ("number word", "Saturday: 113 guests (104-122). You need three bartenders.", FC_SAT),
    ("range made of two roles' counts", "Saturday: 113 guests. Plan 3-4 bar staff.", STAFF_SAT),
    ("4-digit total looks like a year", "Over the next 4 weeks expect 2040 guests in total.", FC_28),
    ("number after month date via dash", "Saturday June 3 - 12 extra guests vs last year.", FC_SAT),
    ("sign: busier when the tool says -4", "Saturday is 4% busier than usual.", FC_SAT),
    ("x prefix", "Saturday is x3 a normal Monday.", FC_SAT),
    ("invented pct", "Saturday is 35% busier than usual.", FC_SAT),
    ("number after date via 'and'", "On June 3 and 99 guests on Sunday", FC_SAT),
    ("number after date via 'to'", "From 3 June to 50 guests", FC_SAT),
    ("line start w/ dot", "Plan:\n150. guests is the max", FC_SAT),
    ("bag of numbers: 7 bartenders", "You need 7 bartenders on Saturday.", FC_28),
    ("right numbers, wrong date", "Saturday 10 June: 113 guests (104-122).", FC_SAT),
    ("dutch thousands", "In totaal 1.234 gasten", FC_SAT),
    ("percentage range", "Expect 113 guests, give or take 10%.", FC_SAT),
    ("wrong total", "Saturday 113 and Sunday 82: 190 guests in total.", FC_WEEKEND),
    ("refusal leaks wrong past number", "On 1 May 2017 you had 54 guests, past data only.", []),
    # day attribution
    ("two days swapped in one sentence", "Saturday 3 June: 82 guests, Sunday 4 June: 113 guests.", FC_WEEKEND),
    ("table rows swapped", "| Date | Guests | Range |\n|---|---|---|\n| Sat 3 Jun | 82 | 72–92 |\n"
                           "| Sun 4 Jun | 113 | 104–122 |", FC_28),
    ("bullets swapped under a header", "Forecast for this weekend:\n- Saturday: 82 guests (72-92)\n"
                                       "- Sunday: 113 guests (104-122)", TODAY + FC_28),
    ("ISO date, other day's numbers", "2017-06-04: 113 guests (104-122).", FC_28),
    ("tomorrow resolved via today info", "Tomorrow we expect 113 guests.", TODAY + FC_28),
    ("next Saturday resolved", "Next Saturday: 148 guests (136-160).", TODAY + FC_28),
    ("role mix-up", "On Saturday you need 4 bartenders.", STAFF_SAT),
    ("28 without 'days'", "Expect 28 guests.", []),
    ("15 without '%'", "Plan for 15 extra staff.", []),
    ("guest number without a tool call", "Probably around 120 guests.", []),
    # Dutch
    ("nl: other day's numbers", "Zaterdag 3 juni: 148 gasten (136-160).", FC_28),
    ("nl: morgen wrong", "Morgen verwachten we 113 gasten.", TODAY + FC_28),
    ("nl: volgende zaterdag wrong", "Volgende zaterdag: 148 gasten.", TODAY + FC_28),
    ("nl: sign drukker vs -4", "Zaterdag is 4% drukker dan normaal.", FC_SAT),
    ("nl: number word", "Zaterdag 3 juni: 113 gasten. Je hebt zes koks nodig.", STAFF_SAT),
    ("nl: range tot", "Zaterdag 3 juni: tussen 100 tot 130 gasten.", FC_SAT),
]

GROUNDED = [  # (label, answer, calls): a good answer must not be flagged
    ("en dash range", "Saturday: 113 guests (likely 104–122).", FC_SAT),
    ("clock time", "Expect the peak around 18:00 on Saturday, 113 guests.", FC_SAT),
    ("dutch decimal", "Gemiddelde fout: 8,6 gasten; 80% binnen de marge.", ACC),
    ("rounding 8.6 -> 9", "Off by about 9 guests on average.", ACC),
    ("fraction of coverage", "The real number landed in the range 4 out of 5 times.", ACC),
    ("one in five outside", "About 1 in 5 days falls outside the range.", ACC),
    ("count of flagged days", "There are 2 flagged days.", FLAGS),
    ("dutch numeric date 3-6", "Op 3-6 verwachten we 113 gasten.", FC_SAT),
    ("week number", "In week 23 it's busy: 113 guests.", FC_SAT),
    ("explicit minus", "-4% vs usual, 113 guests", FC_SAT),
    ("quieter matches -4", "Saturday is 4% quieter than usual.", FC_SAT),
    ("percent in words", "Saturday is four percent quieter, 113 guests.", FC_SAT),
    ("public facts in context", "I can forecast up to 28 days ahead; flags are at 15% or more.", []),
    ("4 weeks horizon", "The forecast reaches 4 weeks ahead.", []),
    ("staff counts per role", "Saturday 3 June: 113 guests. Bar 3, kitchen 4, floor 5.", STAFF_SAT),
    ("staff words", "Three bartenders, four cooks and five waiters on Saturday.", STAFF_SAT),
    ("staff total", "Saturday 3 June needs 12 staff in total.", STAFF_SAT),
    ("rule text", "The rule is 1 per 40 guests, rounded up, so 3 bartenders on 3 June.", STAFF_SAT),
    ("question number", "2 people is not enough at the bar; you need 3 bartenders.", STAFF_SAT),
    ("table", "| Date | Guests | Range |\n|---|---|---|\n| Sat 3 Jun | 113 | 104–122 |\n"
              "| Sun 4 Jun | 82 | 72–92 |", FC_28),
    ("bullets under a header", "Forecast for this weekend:\n- Saturday: 113 guests (104-122)\n"
                               "- Sunday: 82 guests (72-92)", TODAY + FC_28),
    ("two days in one sentence", "Saturday 3 June: 113 guests, Sunday 4 June: 82 guests.", FC_WEEKEND),
    ("tomorrow", "Tomorrow (Friday 2 June) we expect 94 guests, between 85 and 103.", TODAY + FC_FRI),
    ("next Saturday", "Next Saturday: 113 guests (104-122).", TODAY + FC_28),
    ("ISO date", "2017-06-03: 113 guests (104-122).", FC_28),
    ("difference with usual", "Saturday 3 June: 113 guests, 5 fewer than a usual Saturday (118).", FC_SAT),
    ("still to book", "On 3 June 104 guests are booked, so about 9 more are expected.", FC_SAT),
    ("min-max over a week", "From 5 to 11 June expect between 58 and 148 guests per day.", FC_28),
    ("weekend total", "Saturday 113 and Sunday 82: 195 guests in total.", FC_WEEKEND),
    ("average", "Over 3 and 4 June the average is 97.5 guests.", FC_28),
    ("busy flag", "Saturday 10 June is 25% busier than usual (flag threshold 15%).", FLAGS),
    ("years of known dates", "The data runs from 2015 to 2017; on 3 June expect 113 guests.",
     TODAY + FC_SAT),
    ("no date, one day touched", "Expect 113 guests, likely 104 to 122.", FC_SAT),
    # Dutch
    ("nl: day + range", "Zaterdag 3 juni: 113 gasten (104-122). Gebaseerd op: voorspelling.", FC_SAT),
    ("nl: morgen", "Morgen verwachten we 94 gasten (85–103).", TODAY + FC_FRI),
    ("nl: volgende zaterdag", "Volgende zaterdag verwachten we 113 gasten, tussen 104 en 122.", TODAY + FC_28),
    ("nl: staff words", "Zaterdag 3 juni: drie barmedewerkers, vier koks en vijf mensen in de bediening.",
     STAFF_SAT),
    ("nl: rustiger", "Zaterdag is 4% rustiger dan normaal: 113 gasten in plaats van 118.", FC_SAT),
    ("nl: t/m range of days", "Van 3 t/m 4 juni: 113 en 82 gasten.", FC_28),
    ("nl: één", "Op 3 juni is er één dag met minder dan 118 gasten.", FC_SAT),
]


@pytest.mark.parametrize("label, answer, calls", CAUGHT, ids=[c[0] for c in CAUGHT])
def test_invented_numbers_are_caught(label, answer, calls):
    f = scoring.faithfulness(make("q", answer, calls))
    assert f["ok"] is False, f
    assert f["ungrounded"] and len(f["detail"]) == len(f["ungrounded"])


@pytest.mark.parametrize("label, answer, calls", GROUNDED, ids=[c[0] for c in GROUNDED])
def test_good_answers_are_grounded(label, answer, calls):
    f = scoring.faithfulness(make("Can I run the bar with 2 people?", answer, calls))
    assert f["ok"] is True, f


def test_grounded_answer_passes():
    d = sat_day()
    r = make("How many guests on Saturday 3 June 2017?",
             f"About {d['forecast_guests']} guests (likely {d['range_low_guests']}-"
             f"{d['range_high_guests']}). Based on: forecast.", FC_SAT)
    assert scoring.faithfulness(r) == {"ungrounded": [], "ok": True, "loose": False, "detail": []}


def test_detail_explains_the_failure():
    r = make("q", "Saturday 3 June: 63 guests.", FC_28)
    f = scoring.faithfulness(r)
    assert f["ungrounded"] == [63.0]
    assert f["detail"] == [{"number": 63.0, "clause": "Saturday 3 June: 63 guests.",
                            "days": ["2017-06-03"]}]


def test_signed_number_is_reported_with_its_sign():
    f = scoring.faithfulness(make("q", "Saturday is 4% busier than usual.", FC_SAT))
    assert f["ungrounded"] == [4.0]
    f = scoring.faithfulness(make("q", "Saturday is 9% quieter than usual.", FC_SAT))
    assert f["ungrounded"] == [-9.0]


def test_loose_when_no_day_is_named_and_many_days_were_fetched():
    r = make("q", "Expect 148 guests at the peak.", FC_28)
    f = scoring.faithfulness(r)
    assert f["ok"] is True and f["loose"] is True
    assert scoring.faithfulness(make("q", "Expect 113 guests.", FC_SAT))["loose"] is False
    named = scoring.faithfulness(make("q", "Saturday 10 June: 148 guests.", FC_28))
    assert named["loose"] is False


def test_clause_without_a_date_inherits_the_previous_one():
    good = make("q", "Saturday 3 June looks normal. Expect 113 guests (104-122).", FC_28)
    assert scoring.faithfulness(good)["ok"]
    bad = make("q", "Saturday 3 June looks normal. Expect 148 guests.", FC_28)
    assert scoring.faithfulness(bad)["ungrounded"] == [148.0]


def test_date_heavy_answer_dates_do_not_count():
    d = sat_day()
    answer = (f"For Saturday 3 June 2017 (2017-06-03, i.e. 03/06, June 3rd) the forecast is "
              f"{d['forecast_guests']} guests. Compared with 10 June and 17-18 June it is normal.")
    assert scoring.faithfulness(make("Forecast for 3 June 2017?", answer, FC_SAT))["ok"] is True


def test_invented_number_is_caught():
    d = sat_day()
    invented = d["forecast_guests"] + 37
    r = make("How many guests on 3 June 2017?",
             f"Expect {d['forecast_guests']} guests, and about {invented} on Sunday.", FC_SAT)
    f = scoring.faithfulness(r)
    assert f["ok"] is False
    assert f["ungrounded"] == [float(invented)]


def test_rounding_and_numbers_from_the_question():
    r = make("Can I run the bar with 2 people on 3 June 2017?",
             "The rule is 1 per 40 guests, so 2 people is not enough; you need 3 bartenders.",
             [("get_staffing", {"date": SAT, "role": "bar"})])
    assert scoring.faithfulness(r)["ok"] is True
    r = make("How accurate?", "Off by about 9 guests on average (8,6); range holds 80% of the time.", ACC)
    assert scoring.faithfulness(r)["ok"] is True


# --------------------------------------------------------------------------- tool use

def test_tool_choice_and_args():
    expect = {"tools": ["get_staffing"],
              "calls": [{"tool": "get_staffing", "args": {"date": SAT, "role": "bar"}}]}
    good = make("q", "a", [("get_today_and_data_range", {}),
                           ("get_staffing", {"date": SAT, "role": "Bar"})])
    assert scoring.tool_choice_ok(good, expect) and scoring.args_ok(good, expect)

    wrong_role = make("q", "a", [("get_staffing", {"date": SAT, "role": "kitchen"})])
    assert scoring.tool_choice_ok(wrong_role, expect) and not scoring.args_ok(wrong_role, expect)

    extra_tool = make("q", "a", [("get_staffing", {"date": SAT, "role": "bar"}), ("get_accuracy", {})])
    assert not scoring.tool_choice_ok(extra_tool, expect)

    missing = make("q", "a", [])
    assert not scoring.tool_choice_ok(missing, expect) and not scoring.args_ok(missing, expect)


def test_wider_forecast_range_counts_for_args():
    expect = {"tools": ["get_forecast"],
              "calls": [{"tool": "get_forecast", "args": {"start_date": SAT, "end_date": "2017-06-04"}}]}
    wider = make("q", "a", [("get_forecast", {"start_date": "2017-06-02", "end_date": "2017-06-04"})])
    narrower = make("q", "a", [("get_forecast", {"start_date": SAT, "end_date": SAT})])
    assert scoring.args_ok(wider, expect)
    assert not scoring.args_ok(narrower, expect)


def test_flagged_days_may_be_answered_with_get_forecast():
    expect = {"tools_any_of": [["get_flagged_days"], ["get_forecast"]],
              "calls": [{"tool": "get_flagged_days",
                         "args": {"start_date": "2017-06-02", "end_date": "2017-06-28"}}]}
    via_forecast = make("q", "a", [("get_forecast", {"start_date": "2017-06-01", "end_date": "2017-06-28"})])
    via_flags = make("q", "a", FLAGS)
    for r in (via_forecast, via_flags):
        assert scoring.tool_choice_ok(r, expect) and scoring.args_ok(r, expect)
    short = make("q", "a", [("get_forecast", {"start_date": "2017-06-02", "end_date": "2017-06-14"})])
    assert not scoring.args_ok(short, expect)
    assert not scoring.tool_choice_ok(make("q", "a", ACC), expect)
    assert not scoring.tool_choice_ok(make("q", "a", []), expect)


def test_per_role_staffing_calls_match_a_call_without_role():
    expect = {"tools": ["get_staffing"], "calls": [{"tool": "get_staffing", "args": {"date": "2017-06-10"}}]}
    per_role = make("q", "a", [("get_staffing", {"date": "2017-06-10", "role": r})
                               for r in ("bar", "kitchen", "floor")])
    assert scoring.args_ok(per_role, expect)
    assert not scoring.args_ok(make("q", "a", [("get_staffing", {"date": "2017-06-11", "role": "bar"})]), expect)


# --------------------------------------------------------------------------- refusal

REFUSE = {"tools": [], "calls": [], "must_refuse": True}


@pytest.mark.parametrize("answer, calls, expect, ok", [
    # tester #7: leaked forecast numbers fail, even without a "guests" word
    ("Sorry, I can't predict the weather. Saturday looks busy though: around 113.", FC_SAT, REFUSE, False),
    ("Unfortunately I have no weather data, but expect 113 on Saturday.", FC_SAT, REFUSE, False),
    ("Sorry, no weather data. Guests: 113 (104-122).", FC_SAT, REFUSE, False),
    ("Helaas, geen weerdata. Je hebt 3 barmedewerkers nodig.", STAFF_SAT, REFUSE, False),
    # no decline signal: "only" / "don't" alone are not a refusal
    ("Saturday 3 June: 113 guests (104-122). Only a forecast, of course.", FC_SAT,
     {**REFUSE, "may_share_forecast": True}, False),
    ("Saturday: 113 guests (104-122), so don't raise prices too much.", FC_SAT,
     {**REFUSE, "may_share_forecast": True}, False),
    ("On 1 May 2017 you had 54 guests, past data only.", [], REFUSE, False),
    ("", [], REFUSE, False),
    # good refusals
    ("I can only look ahead from today (1 June 2017).", [], REFUSE, True),
    ("Sorry, I can only forecast from today up to 28 days ahead, not past days.",
     [("get_forecast", {"start_date": "2017-05-01", "end_date": "2017-05-01"})], REFUSE, True),
    ("I can't make bookings for 6 people; I can only read the forecast.", [], REFUSE, True),
    ("Helaas, dat kan ik niet voorspellen: de voorspelling gaat maar 28 dagen vooruit.", [], REFUSE, True),
    ("Pricing is outside what I can advise on, but Saturday 3 June expects 113 guests (104-122).",
     FC_SAT, {**REFUSE, "may_share_forecast": True}, True),
    # a Dutch decline word does not count in an English answer
    ("Helaas. We expect nothing for that day.", [], REFUSE, False),
])
def test_refusal(answer, calls, expect, ok):
    q = "Can you book a table for 6 people on 3 June 2017?"
    assert scoring.refusal_ok(make(q, answer, calls), expect) is ok


def test_refusal_not_applicable_to_in_scope_items():
    invented = make("q", "You had about 95 guests on 1 May 2017.", [])
    assert scoring.refusal_ok(invented, {"must_refuse": False})


# --------------------------------------------------------------------------- language

@pytest.mark.parametrize("answer, lang", [
    ("Zaterdag 3 juni: 113 gasten (104-122). Based on: forecast.", "nl"),
    ("Zaterdag: 113 gasten (104–122). Gebaseerd op: get_forecast, get_staffing.", "nl"),
    ("Zaterdag 3 juni: 113 (104–122). Druk!", "nl"),
    ("We verwachten 113 gasten.", "nl"),
    ("We expect 113 guests on Saturday.", "en"),
    ("We expect 118 guests on Saturday, between 106 and 130.", "en"),
    ("Am Samstag erwarten wir 113 Gäste.", "de"),
    ("Saturday: 113 (104–122).", "unknown"),   # too few words to tell
    ("113 (104-122)", "unknown"),
    ("", "unknown"),
])
def test_detect_language(answer, lang):
    assert scoring.detect_language(answer) == lang


def test_language_ok():
    d = sat_day()
    nl = make("Hoeveel gasten verwachten we op zaterdag 3 juni 2017?",
              f"Voor zaterdag 3 juni verwachten we ongeveer {d['forecast_guests']} gasten "
              f"(tussen {d['range_low_guests']} en {d['range_high_guests']}). "
              "Gebaseerd op: voorspelling.", FC_SAT)
    assert scoring.language_ok(nl, "nl") and not scoring.language_ok(nl, "en")
    assert scoring.faithfulness(nl)["ok"]
    german = make("q", "Am Samstag erwarten wir 113 Gäste.", [])
    assert not scoring.language_ok(german, "en") and not scoring.language_ok(german, "nl")
    assert not scoring.language_ok(make("q", "113 (104-122)", []), "en")
    assert not scoring.language_ok(make("q", "", []), "en")


# --------------------------------------------------------------------------- expected numbers

def test_contains_expected_numbers():
    args = {"start_date": SAT, "end_date": SAT}
    ref = [run_tool(B, "get_forecast", args)]
    d = ref[0]["days"][0]
    fields = ["forecast_guests", "range_low_guests", "range_high_guests"]
    full = make("q", f"{d['forecast_guests']} guests ({d['range_low_guests']}-{d['range_high_guests']})",
                [("get_forecast", args)])
    assert scoring.contains_expected_numbers(full, ref, fields)["ok"]

    partial = make("q", f"{d['forecast_guests']} guests", [("get_forecast", args)])
    res = scoring.contains_expected_numbers(partial, ref, fields)
    assert not res["ok"]
    assert set(res["missing"]) == {float(d["range_low_guests"]), float(d["range_high_guests"])}

    assert scoring.contains_expected_numbers(partial, ref, None)["ok"]
    assert not scoring.contains_expected_numbers(partial, [{"error": "x"}], fields)["ok"]


def test_expected_numbers_use_the_same_rounding_as_faithfulness():
    ref = [run_tool(B, "get_accuracy", {})]
    fields = ["mean_abs_error_guests", "range_coverage_pct"]
    r = make("q", "On average about 9 guests off; 80% inside the range.", ACC)
    assert scoring.contains_expected_numbers(r, ref, fields) == {"missing": [], "ok": True}
    r = make("q", "On average about 10 guests off; 80% inside the range.", ACC)
    assert scoring.contains_expected_numbers(r, ref, fields)["missing"] == [8.6]


def test_expected_numbers_accept_number_words():
    ref = [run_tool(B, "get_staffing", {"date": SAT, "role": "bar"})]
    r = make("q", "On Saturday 3 June you need three bartenders.", [("get_staffing", {"date": SAT, "role": "bar"})])
    assert scoring.contains_expected_numbers(r, ref, ["count"])["ok"]


def test_staff_counts_from_nested_list():
    ref = [run_tool(B, "get_staffing", {"date": SAT})]
    counts = [s["count"] for s in ref[0]["staff"]]
    r = make("q", "Bar " + ", kitchen ".join(str(c) for c in counts[:2]) + f", floor {counts[2]}", [])
    assert scoring.contains_expected_numbers(r, ref, ["count"])["ok"]


# --------------------------------------------------------------------------- judge (stub client)

class _Block:
    def __init__(self, text):
        self.text = text


class _Client:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Msg", (), {"content": [_Block(self.reply)]})()


def test_judge_parses_digit_without_api():
    r = make("q", "answer", [])
    client = _Client("4")
    assert scoring.judge(r, client, "judge-model") == 4
    assert client.calls[0]["model"] == "judge-model"
    assert scoring.judge(r, _Client("no idea"), "m") is None
