"""Unit tests for evals/scoring.py with hand-made RunResults (no API).

The faithfulness tables include every case from the independent tester's probes
(score_probe.py / score_probe2.py / other_probe.py), with the FakeBackend's current numbers.
FakeBackend facts used below (today = Thursday 2017-06-01):
  Sat 2017-06-03: booked 104, usual 118, forecast 113, range 104-122, vs usual -4%
  Sun 2017-06-04: forecast 82, range 72-92        Fri 2017-06-02: forecast 94, range 85-103
  Sat 2017-06-10: forecast 148, range 136-160, +25%, busy
  Staffing 2017-06-03: bar 3, kitchen 4, floor 5 (1 per 40 / 30 / 25 guests)
  Sat 2017-06-10 busy (+25%), Tue 2017-06-20 quiet (-15%): the only 2 flagged days
  Accuracy: backtest mean abs error 4.2 / 5.9 / 8.6 / 11.3 guests (1/3/7/14 days ahead),
  range_percentile 80 ("8 in 10 past days stayed within"), after export 69% inside the range
"""
from __future__ import annotations

import pytest

from agent.contract import FLAG_THRESHOLD_PCT, RunResult, ToolCall, run_tool
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
    ("dutch decimal", "Gemiddelde fout: 8,6 gasten; 69% binnen de marge.", ACC),
    ("rounding 8.6 -> 9", "Off by about 9 guests on average.", ACC),
    ("coverage pct", "After the export the real number landed inside the range on 69% of days.", ACC),
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
    r = make("How accurate?", "Off by about 9 guests on average (8,6); range holds 69% of the time.", ACC)
    assert scoring.faithfulness(r)["ok"] is True


# --------------------------------------------------------------------------- second tester round
# Hand-made tool outputs copied from the real backend (today 2017-06-01), so these cases do not
# depend on the data file: Sat 24 Jun is -14.5% (NOT flagged), Wed 7 Jun is -15.8% (flagged).

def _day(date, weekday, booked, usual, forecast, lo, hi, pct, flag):
    return {"date": date, "weekday": weekday, "booked_so_far_guests": booked,
            "usual_for_weekday_guests": usual, "forecast_guests": forecast, "range_low_guests": lo,
            "range_high_guests": hi, "vs_usual_pct": pct, "flag": flag, "closed": False}


REAL_SAT24 = _day("2017-06-24", "Saturday", 106, 136, 116, 100, 132, -14.5, None)
REAL_WED7 = _day("2017-06-07", "Wednesday", 98, 127, 107, 95, 118, -15.8, "quiet")
REAL_ACC = {"backtest": {"weeks": 8, "description": "Replaying the 8 weeks before the export.",
                         "by_days_ahead": [{"days_ahead": 7, "mean_abs_error_guests": 7.9,
                                            "weekday_average_mean_abs_error_guests": 26.7,
                                            "range_half_width_guests": 11.7}]},
            "range_percentile": 80,
            "range_meaning": "8 in 10 past days stayed within this distance of the forecast",
            "after_export": {"description": "What really happened on the next 14 days.",
                             "days_checked": 14, "inside_range_pct": 64,
                             "mean_abs_error_guests": 8.9,
                             "weekday_average_mean_abs_error_guests": 19.0}}


def _out(name: str, output: dict) -> ToolCall:
    """A tool call with a hand-made output (no backend)."""
    return ToolCall(name=name, input={}, output=output, seconds=0.0)


def make_any(answer: str, calls: list) -> RunResult:
    """Like make(), but a ToolCall in ``calls`` is used as it is."""
    r = make("q", answer, [c for c in calls if not isinstance(c, ToolCall)])
    r.tool_calls = [c for c in calls if isinstance(c, ToolCall)] + r.tool_calls
    return r


SAT24 = [_out("get_forecast", {"days": [REAL_SAT24]})]
WED7 = [_out("get_flagged_days", {"days": [REAL_WED7], "threshold_pct": FLAG_THRESHOLD_PCT})]
RACC = [_out("get_accuracy", REAL_ACC)]

ROUND2 = [  # (label, answer, outputs, grounded?)
    # 1. 15 is the flag threshold, not a day's change; rounding must not cross the threshold
    ("15% for -14.5 makes it look flagged", "Sat 24 Jun looks 15% quieter than usual.", SAT24, False),
    ("-15% for -14.5", "Sat 24 Jun: -15% vs usual.", SAT24, False),
    ("15% quieter even with 'flagged' nearby", "Sat 24 Jun is 15% quieter, so it is flagged.", SAT24, False),
    ("threshold_pct in the output is no day value", "Wed 7 Jun is 15% quieter than usual.",
     WED7 + SAT24, False),
    ("-14.5 written exactly", "Sat 24 Jun: -14.5% vs usual, just under the 15% flag.", SAT24, True),
    ("14.5 quieter", "Sat 24 Jun looks 14.5% quieter than usual, below the threshold.", SAT24, True),
    ("threshold: at least 15%", "Days at least 15% busier or quieter than usual are flagged.", [], True),
    ("threshold: 15% or more", "A day is flagged at 15% or more.", [], True),
    ("threshold: drempel", "De drempel is 15%; zaterdag 24 juni blijft eronder.", SAT24, True),
    ("window: 28 days", "I can only look 28 days ahead.", [], True),
    ("window: 4 weeks", "The forecast reaches 4 weeks ahead.", [], True),
    ("flagged -15.8 rounded to 16", "Wed 7 Jun: 16% quieter than usual (107 guests).", WED7, True),
    ("flagged -15.8 rounded to 15 is not ok", "Wed 7 Jun is 15% quieter than usual.", WED7, False),
    # 2. range_percentile (80) defines the range; it is not how often the range held
    ("80% coverage claim", "The likely range held the real number on 80% of days.", RACC, False),
    ("80% landed inside", "80% of real numbers landed inside the range.", RACC, False),
    ("8 in 10 as coverage", "The range held the real number on 8 in 10 days.", RACC, False),
    ("nl: 80% lag binnen", "Het echte aantal lag op 80% van de dagen binnen de marge.", RACC, False),
    ("definition: 8 in 10", "The range is set so 8 in 10 past days stayed within it.", RACC, True),
    ("definition: 80% stayed within", "80% of past days stayed within this distance.", RACC, True),
    ("definition: 80th percentile", "The range is the 80th percentile of past errors.", RACC, True),
    ("real coverage 64%", "After the export the range held the real number on 64% of 14 days.", RACC, True),
    # 4. counts of flagged days and averages
    ("3 days flagged when 2 are", "3 days are flagged.", FLAGS, False),
    ("2 flagged days", "2 days are flagged in the next 4 weeks.", FLAGS, True),
    ("1 busy day", "There is 1 busy day: Sat 10 Jun (+25%).", FLAGS, True),
    ("2 busy days when 1 is", "There are 2 busy days.", FLAGS, False),
    ("count of days the answer lists", "Two busy days: Sat 3 Jun and Sun 4 Jun.", FC_WEEKEND, True),
    ("window is not a count", "In the next 28 days, 2 days are flagged.", FLAGS, True),
    ("loose average from one day's number", "On average 113 guests a day over the next two weeks.",
     FC_28, False),
    ("loose average = usual value", "On average a Saturday gets 118 guests.", FC_28, True),
]


@pytest.mark.parametrize("label, answer, calls, grounded", ROUND2, ids=[c[0] for c in ROUND2])
def test_second_tester_round(label, answer, calls, grounded):
    r = make_any(answer, calls)
    assert scoring.faithfulness(r)["ok"] is grounded, scoring.faithfulness(r)


@pytest.mark.parametrize("answer, lang", [
    ("Vrijdag 16 juni: 4 koks.", "nl"),
    ("vr 16 jun: personeel nodig: kitchen 4 (voor 132 gasten verwacht). "
     "Regel: 1 per 35 guests (on the forecast), rounded up, minimum 2.\nGebaseerd op: voorspelling.", "nl"),
    ("za 10 jun: bar 2 nodig (voor 74 gasten verwacht).", "nl"),
    ("Zaterdag 3 juni: bar 3, keuken 4.", "nl"),
    ("Sat 3 Jun: staff needed: bar 3 (for 138 guests expected). "
     "Rule: 1 per 60 guests (on the forecast), rounded up, minimum 1.", "en"),
    ("Friday 16 June: 4 cooks.", "en"),
])
def test_detect_language_short_and_tool_text(answer, lang):
    assert scoring.detect_language(answer) == lang


def test_known_limits_are_documented():
    assert "KNOWN_LIMITS" in scoring.__dict__ and scoring.KNOWN_LIMITS.strip()


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
    ("Saturday: 113 (104–122).", "en"),        # too few words: the English day name decides
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
    assert scoring.contains_expected_numbers(
        make("q", "After the export, 69% landed inside the range.", ACC), ref,
        ["inside_range_pct"]) == {"missing": [], "ok": True}
    assert scoring.contains_expected_numbers(
        make("q", "After the export, 80% landed inside the range.", ACC), ref,
        ["inside_range_pct"])["missing"] == [69.0]
    # 4.2 -> 4, 5.9 -> 6, 8.6 -> 9, 11.3 -> 11, 9.4 -> 9
    r = make("q", "Off by 4, 6, 9 and 11 guests at 1, 3, 7 and 14 days ahead; 69% inside.", ACC)
    assert scoring.contains_expected_numbers(r, ref, ["mean_abs_error_guests", "inside_range_pct"])["ok"]
    r = make("q", "Off by 4, 6, 10 and 11 guests at 1, 3, 7 and 14 days ahead; 69% inside.", ACC)
    assert scoring.contains_expected_numbers(r, ref, ["mean_abs_error_guests"])["missing"] == [8.6, 9.4]


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


# --------------------------------------------------------------------------- first live eval
# Real model answers from the first live eval in CI, with the tool outputs copied from
# RealBackend (today Thu 2017-06-01). test_live_outputs_match_the_real_backend checks the copies.

REAL_TODAY = {"today": "2017-06-01", "first_booking_date": "2015-07-01",
              "last_booking_date": "2017-08-31",
              "dataset": "City hotel sample (real bookings, export of 31 May 2017)",
              "max_horizon_days": 28, "today_weekday": "Thursday", "last_forecast_date": "2017-06-28"}
REAL_SAT17 = _day("2017-06-17", "Saturday", 154, 136, 157, 141, 173, 15.9, "busy")
REAL_STAFF17 = {"date": "2017-06-17", "forecast_guests": 157, "staff": [
    {"role": "floor", "count": 7, "rule": "1 per 25 guests (on the forecast), rounded up, minimum 2"},
    {"role": "bar", "count": 3, "rule": "1 per 60 guests (on the forecast), rounded up, minimum 1"},
    {"role": "kitchen", "count": 5, "rule": "1 per 35 guests (on the forecast), rounded up, minimum 2"}]}
REAL_ACC_FULL = {
    "backtest": {"weeks": 8, "description": (
        "Replaying the 8 weeks before the export: for each past day the forecast is rebuilt from "
        "the bookings known back then and compared with what really came, per number of days ahead."),
        "by_days_ahead": [
            {"days_ahead": 1, "mean_abs_error_guests": 5.1, "weekday_average_mean_abs_error_guests": 26.7,
             "range_half_width_guests": 6.2},
            {"days_ahead": 3, "mean_abs_error_guests": 6.8, "weekday_average_mean_abs_error_guests": 26.7,
             "range_half_width_guests": 8.8},
            {"days_ahead": 7, "mean_abs_error_guests": 7.9, "weekday_average_mean_abs_error_guests": 26.7,
             "range_half_width_guests": 11.7},
            {"days_ahead": 14, "mean_abs_error_guests": 10.6, "weekday_average_mean_abs_error_guests": 26.5,
             "range_half_width_guests": 16.0}]},
    "range_percentile": 80,
    "range_meaning": "8 in 10 past days stayed within this distance of the forecast",
    "after_export": {"description": "Known only because the sample is history: what really happened "
                                    "on the next 14 days after the export.",
                     "days_checked": 14, "inside_range_pct": 64, "mean_abs_error_guests": 8.9,
                     "weekday_average_mean_abs_error_guests": 19.0}}
REAL_22_25 = [_day("2017-06-22", "Thursday", 134, 153, 144, 128, 160, -6.0, None),
              _day("2017-06-23", "Friday", 135, 173, 139, 123, 155, -19.8, "quiet"),
              REAL_SAT24,
              _day("2017-06-25", "Sunday", 185, 123, 184, 168, 200, 50.0, "busy")]

_D17 = {"start_date": "2017-06-17", "end_date": "2017-06-17"}
_D22_25 = {"start_date": "2017-06-22", "end_date": "2017-06-25"}
LIVE_CALLS = {  # (tool, args) -> real output
    ("get_today_and_data_range", ()): REAL_TODAY,
    ("get_staffing", (("date", "2017-06-17"),)): REAL_STAFF17,
    ("get_forecast", tuple(_D17.items())): {"days": [REAL_SAT17]},
    ("get_accuracy", ()): REAL_ACC_FULL,
    ("get_forecast", tuple(_D22_25.items())): {"days": REAL_22_25},
}


def live(question: str, answer: str, calls: list[tuple[str, dict]]) -> RunResult:
    tool_calls = [ToolCall(n, a, LIVE_CALLS[(n, tuple(a.items()))], 0.0) for n, a in calls]
    return RunResult(question=question, answer=answer, model="live", prompt_version="v1",
                     tool_calls=tool_calls, stop_reason="answered")


def test_live_outputs_match_the_real_backend():
    pytest.importorskip("pace.plan")
    from agent.real_backend import RealBackend
    real = RealBackend()
    for (name, args), output in LIVE_CALLS.items():
        assert run_tool(real, name, dict(args)) == output, name


ST08_Q = "How did you work out how many floor staff I need on 17 June 2017?"
ST08_A = ("On Sat 17 Jun, the plan is **7 floor staff**. Here's how that was worked out:\n\n"
          "- **Guest forecast:** 157 guests (likely range 141-173). Already booked: 154. This day is "
          "flagged busy, about 16% above the usual Saturday.\n"
          "- **Floor rule:** 1 floor staff per 25 forecast guests, rounded up, with a minimum of 2.\n\n"
          "The rule applies to the 157 forecast, not the full range. I can't check the staffing at the "
          "top or bottom of the range with the tools I have.\n\nBased on: staffing rules, forecast.")
ST08_CALLS = [("get_today_and_data_range", {}), ("get_staffing", {"date": "2017-06-17"}),
              ("get_forecast", _D17)]
ST08_EXPECT = {"tools": ["get_staffing"],
               "calls": [{"tool": "get_staffing", "args": {"date": "2017-06-17", "role": "floor"}}],
               "numbers_from": ["count"]}


def test_live_st08_all_roles_call_and_forecast_are_fine():
    """st08 was a scorer false alarm: get_staffing without a role returns floor too, and a
    staffing answer may call get_forecast for the range and the explanation."""
    r = live(ST08_Q, ST08_A, ST08_CALLS)
    assert scoring.tool_choice_ok(r, ST08_EXPECT)
    assert scoring.args_ok(r, ST08_EXPECT)
    assert scoring.faithfulness(r)["ok"], scoring.faithfulness(r)


def test_staffing_without_role_matches_every_role_but_not_other_days_or_tools():
    for role in ("floor", "bar", "kitchen"):
        expect = {"tools": ["get_staffing"],
                  "calls": [{"tool": "get_staffing", "args": {"date": SAT, "role": role}}]}
        assert scoring.args_ok(make("q", "a", STAFF_SAT), expect)
        assert scoring.tool_choice_ok(make("q", "a", STAFF_SAT + FC_SAT), expect)
        assert not scoring.args_ok(make("q", "a", [("get_staffing", {"date": "2017-06-04"})]), expect)
        assert not scoring.tool_choice_ok(make("q", "a", STAFF_SAT + ACC), expect)
    # a forecast-only question still may not call get_staffing
    fc_only = {"tools": ["get_forecast"], "calls": [{"tool": "get_forecast", "args": {
        "start_date": SAT, "end_date": SAT}}]}
    assert not scoring.tool_choice_ok(make("q", "a", FC_SAT + STAFF_SAT), fc_only)


TA02_Q = "How sure are you about the forecast for Saturday 17 June 2017?"
TA02_A = ("**Saturday 17 Jun: forecast 157 guests, likely range 141-173.** It's flagged busy, against "
          "a usual Saturday of 136. So far 154 guests are booked.\n\nHow sure I am:\n"
          "- **Accuracy:** At 14 days ahead, the typical error in past backtests was about 11 guests. "
          "At 1 to 7 days it was 5 to 8. The 17th is 16 days away, so it falls just beyond the range "
          "I tested.\n"
          "- **Range reliability:** The range is meant to hold 8 in 10 days. In a later check on the "
          "14 days after the export, only 64% landed inside it. Treat the range as a bit optimistic, "
          "and expect the forecast to tighten as the date gets closer.\n\n"
          "Based on: forecast, accuracy check.")
TA02_CALLS = [("get_today_and_data_range", {}), ("get_accuracy", {}), ("get_forecast", _D17)]


def test_live_ta02_accuracy_answer_is_grounded():
    """ta02 was a scorer false alarm: ranges of undated accuracy values ("1 to 7", "5 to 8"),
    the range definition as a rate ("meant to hold 8 in 10 days") and a day's distance."""
    r = live(TA02_Q, TA02_A, TA02_CALLS)
    assert scoring.faithfulness(r)["ok"], scoring.faithfulness(r)


@pytest.mark.parametrize("label, answer, grounded", [
    ("undated range in a dated clause", "Sat 17 Jun: at 1 to 7 days ahead the error was 5 to 8.", True),
    ("undated range, wrong end", "Sat 17 Jun: at 1 to 7 days ahead the error was 5 to 12.", False),
    ("8 in 10 definition", "Sat 17 Jun: the range is meant to hold 8 in 10 days.", True),
    ("8 out of 10", "The likely range is built from 8 out of 10 past days.", True),
    ("8 op de 10", "De marge is zo gekozen dat 8 op de 10 dagen erbinnen blijven.", True),
    ("8 in 10 as what happened", "After the export the range held the real number on 8 in 10 days.",
     False),
    ("distance from today", "Sat 24 Jun is 23 days away.", True),
    ("wrong distance", "Sat 24 Jun is 25 days away.", False),
    ("distance of another day", "Sat 17 Jun is 23 days away.", False),
])
def test_live_ta02_rules(label, answer, grounded):
    calls = TA02_CALLS + [("get_forecast", _D22_25)]
    r = live(TA02_Q, answer, calls)
    assert scoring.faithfulness(r)["ok"] is grounded, scoring.faithfulness(r)


DR07_Q = "How do 22 to 25 June 2017 compare with what is usual for those weekdays?"
DR07_CALLS = [("get_today_and_data_range", {}), ("get_forecast", _D22_25)]


@pytest.mark.parametrize("line, grounded", [
    # the real model mistake: -14.5 (not flagged) rounded to 15 sounds flagged
    ("- **Sat 24 Jun:** forecast 116 (likely 100–132), usual 136, about 15% below.", False),
    ("- **Sat 24 Jun:** forecast 116 (likely 100–132), usual 136, 14.5% below.", True),
    ("- **Thu 22 Jun:** forecast 144 (likely 128–160), usual 153, about 6% below.", True),
    ("- **Fri 23 Jun:** forecast 139 (likely 123–155), usual 173, about 20% below.", True),
    ("- **Sun 25 Jun:** forecast 184 (likely 168–200), usual 123, about 50% above.", True),
])
def test_live_dr07_rounding_across_the_flag_is_caught(line, grounded):
    r = live(DR07_Q, line, DR07_CALLS)
    assert scoring.faithfulness(r)["ok"] is grounded, scoring.faithfulness(r)


def test_live_dr07_whole_answer_fails_only_on_sat_24():
    answer = "\n".join([
        "Here is how 22 to 25 June compare with usual:",
        "- **Thu 22 Jun:** forecast 144 (likely 128–160), usual 153, about 6% below.",
        "- **Fri 23 Jun:** forecast 139 (likely 123–155), usual 173, about 20% below (flagged quiet).",
        "- **Sat 24 Jun:** forecast 116 (likely 100–132), usual 136, about 15% below.",
        "- **Sun 25 Jun:** forecast 184 (likely 168–200), usual 123, about 50% above (flagged busy).",
        "", "Based on: forecast."])
    f = scoring.faithfulness(live(DR07_Q, answer, DR07_CALLS))
    assert f["ungrounded"] == [-15.0], f
    assert f["detail"][0]["days"] == ["2017-06-24"]


# --------------------------------------------------------------------------- full live run (v1)
# One small case per rule that the 22 failures of the full v1 live run led to (the cases
# themselves are in tests/test_live_eval_v1.py).

LIVE_RULES = [  # (label, answer, calls, grounded?)
    ("Dutch 'ten opzichte van' is not 10", "Za 3 jun: niet afwijkend ten opzichte van normaal (-4%).",
     FC_SAT, True),
    ("English 'ten' is still 10", "Sat 3 Jun: ten more guests than usual.", FC_SAT, False),
    ("direction word on the next line is not this number's",
     "**Busier:**\n- Sat 3 Jun: 113 guests, usual 118\n\n**Quieter:**\n- none", FC_SAT, True),
    ("bullet dash is not a range joiner",
     "- 1 day ahead: 5.1, range ±6.2\n- 3 days ahead: 6.8, range ±8.8",
     [_out("get_accuracy", REAL_ACC_FULL)], True),
    ("'respectively' uses the days named before",
     "- Sat 3 Jun: forecast 113\n- Sun 4 Jun: forecast 82\nBooked so far: 104 and 71 respectively.",
     FC_WEEKEND, True),
    ("'respectively' with a wrong number",
     "- Sat 3 Jun: forecast 113\n- Sun 4 Jun: forecast 82\nBooked so far: 99 and 71 respectively.",
     FC_WEEKEND, False),
    ("'the 80% range'", "In backtests the 80% range was ±11.7 guests.", RACC, True),
    ("'below the 80% target' next to a coverage claim",
     "Only 64% landed inside the range, below the 80% target.", RACC, True),
    ("nl: 'de beoogde 80%'", "Er viel 64% binnen de bandbreedte, minder dan de beoogde 80%.", RACC, True),
    ("80% as what happened still fails", "80% of real numbers landed inside the range.", RACC, False),
    ("weeks away = distance / 7", "Sat 17 Jun is about two weeks out.",
     [_out("get_today_and_data_range", REAL_TODAY), _out("get_forecast", {"days": [REAL_SAT17]})], True),
    ("wrong weeks away", "Sat 17 Jun is about five weeks out.",
     [_out("get_today_and_data_range", REAL_TODAY), _out("get_forecast", {"days": [REAL_SAT17]})], False),
]


@pytest.mark.parametrize("label, answer, calls, grounded", LIVE_RULES, ids=[c[0] for c in LIVE_RULES])
def test_full_live_run_rules(label, answer, calls, grounded):
    r = make_any(answer, calls)
    assert scoring.faithfulness(r)["ok"] is grounded, scoring.faithfulness(r)


def test_dutch_decline_with_words_between_kan_and_geen():
    expect = {"must_refuse": True}
    r = make("Hoeveel gasten verwachten we op 15 juli 2017?",
             "Ik kan voor za 15 jul 2017 geen gastenaantal geven. Mijn voorspelling reikt tot 28 jun.",
             TODAY)
    assert scoring.refusal_ok(r, expect, "nl")
    assert not scoring.refusal_ok(make("q", "Ik kan je helpen: za 15 jul wordt druk.", TODAY), expect, "nl")


# --------------------------------------------------------------------------- "3 of the 4 days"
# From the live eval of 17 Nov (dr07): a correct count that the scorer used to flag, because
# "three of the four" was read as a rate ("3 in 4") instead of a count of days.

def _jun(day, forecast, usual, pct, flag=None):
    return {"date": f"2017-06-{day}", "weekday": "x", "booked_so_far_guests": 100,
            "usual_for_weekday_guests": usual, "forecast_guests": forecast,
            "range_low_guests": forecast - 16, "range_high_guests": forecast + 16,
            "vs_usual_pct": pct, "flag": flag, "closed": False}


DR07_DAYS = [_out("get_forecast", {"days": [_jun(22, 144, 153, -6.0), _jun(23, 139, 173, -19.8, "quiet"),
                                            _jun(24, 116, 136, -14.5), _jun(25, 184, 123, 50.0, "busy")]})]


@pytest.mark.parametrize("answer, grounded", [
    ("Three of the four days are forecast below usual and Sunday is far above.", True),
    ("Drie van de vier dagen liggen onder het gebruikelijke aantal.", True),
    ("One of the four days is above usual.", True),
    ("Two of the four days are flagged.", True),
    ("Two of the four days are below usual.", False),   # it is three
    ("Three of the five days are below usual.", False),  # four days were returned
    ("Three of the four days are closed.", False),       # none is
])
def test_part_of_whole_day_counts(answer, grounded):
    assert scoring.faithfulness(make_any(answer, DR07_DAYS))["ok"] is grounded, answer


@pytest.mark.parametrize("answer", [
    "Three of the four days are above usual, Sunday is below.",                   # states swapped
    "Only one of the four days is below usual, and three of the four days are above it.",
])
def test_part_of_whole_count_must_match_the_state_named_next_to_it(answer):
    assert scoring.faithfulness(make_any(answer, DR07_DAYS))["ok"] is False, answer


def test_part_and_whole_come_from_the_same_call():
    week = [_jun(d, 100, 120, -16.7) for d in range(10, 15)] + [_jun(d, 130, 120, 8.3) for d in (15, 16)]
    calls = DR07_DAYS + [_out("get_forecast", {"days": week})]  # 4 days (3 below) and 7 days (5 below)
    assert scoring.faithfulness(make_any("Five of the seven days are below usual.", calls))["ok"]
    assert not scoring.faithfulness(make_any("Three of the seven days are below usual.", calls))["ok"]


def test_eight_in_ten_past_days_within_the_range_is_the_definition():
    """ta02, live run of 19 Nov: the tool's own wording of the 80% range, not a coverage claim."""
    answer = ("At 14 days ahead, the average error was 8.9 guests, and 8 in 10 past days landed "
              "within ±16.0 guests of the forecast.")
    acc = _out("get_accuracy", {"backtest": {"by_days_ahead": [
        {"days_ahead": 14, "mean_abs_error_guests": 8.9, "range_half_width_guests": 16.0}]},
        "range_percentile": 80, "range_meaning": "8 in 10 past days stayed within this distance of the forecast"})
    assert scoring.faithfulness(make_any(answer, [acc]))["ok"]
    for claim in ["Afterwards, 8 in 10 actual results landed inside the range.",
                  "Over the past days since the export, 8 in 10 landed within the range.",
                  "In the 14 days after the export, 8 in 10 of those past days landed within ±16 guests.",
                  "Of the previous days, 80% landed within the likely range."]:
        assert not scoring.faithfulness(make_any(claim, [acc]))["ok"], claim  # claims to check
