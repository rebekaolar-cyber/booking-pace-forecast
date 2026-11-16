"""RealBackend: the assistant's tools on the real forecast core, checked against the app's own logic."""
import pytest

pd = pytest.importorskip("pandas")
P = pytest.importorskip("pace.plan")

import datetime as dt  # noqa: E402

from agent.contract import MAX_HORIZON_DAYS, TOOL_NAMES, run_tool  # noqa: E402
from agent.fake_backend import FakeBackend  # noqa: E402
from agent.real_backend import RealBackend  # noqa: E402

TODAY = "2017-06-01"
LAST = "2017-06-28"
SAT = "2017-06-03"


@pytest.fixture(scope="module")
def real():
    return RealBackend()


def app_run_plan(raw, until, days=MAX_HORIZON_DAYS, skip_closed=True, actual=None):
    """streamlit_app.py's run_plan, copied line for line (it can't be imported: it starts Streamlit)."""
    prep = P.prepare(raw, P.guess_columns(raw.columns), True)
    prep = P.Prepared(prep.reservations, dropped=0)
    check = P.self_check(prep, until, actual, skip_closed=skip_closed)
    book = P.build_book(prep, until, actual)
    typical = dict(zip(check.days_ahead, check.pace_p80)) if len(check) else None
    return P.plan(book, until, days=days, skip_closed=skip_closed, typical_error=typical), check


@pytest.fixture(scope="module")
def app_plan():
    p, _ = app_run_plan(P.sample_export(), P.SAMPLE_EXPORT_DAY)
    return p.set_index("date", drop=False)


def test_today_is_the_day_after_the_export(real):
    info = run_tool(real, "get_today_and_data_range", {})
    assert info["today"] == TODAY and info["last_forecast_date"] == LAST
    assert info["dataset"].startswith("City hotel sample")
    assert info["first_booking_date"] <= "2015-07-01" and info["last_booking_date"] > LAST


def test_every_tool_works_inside_the_window(real):
    args = {"get_today_and_data_range": {}, "get_accuracy": {},
            "get_forecast": {"start_date": TODAY, "end_date": LAST},
            "get_flagged_days": {"start_date": TODAY, "end_date": LAST},
            "get_staffing": {"date": SAT}}
    assert set(args) == set(TOOL_NAMES)
    for name, a in args.items():
        out = run_tool(real, name, a)
        assert "error" not in out, (name, out)
    assert len(run_tool(real, "get_forecast", args["get_forecast"])["days"]) == 28
    assert "error" not in run_tool(real, "get_staffing", {"date": LAST, "role": "Bar"})


def test_past_and_too_far_dates_get_the_contract_errors(real):
    past = run_tool(real, "get_forecast", {"start_date": "2017-05-31", "end_date": "2017-06-02"})
    assert past["error"].startswith(f"start_date 2017-05-31 is in the past; the forecast covers {TODAY} to {LAST}")
    far = run_tool(real, "get_forecast", {"start_date": TODAY, "end_date": "2017-06-29"})
    assert far["error"] == f"The forecast only reaches 28 days ahead, up to {LAST}."
    assert "in the past" in run_tool(real, "get_staffing", {"date": "2017-05-31"})["error"]
    assert "28 days ahead" in run_tool(real, "get_staffing", {"date": "2017-06-29"})["error"]
    assert "Unknown role" in run_tool(real, "get_staffing", {"date": SAT, "role": "dj"})["error"]


@pytest.mark.parametrize("day", ["2017-06-01", SAT, "2017-06-17"])
def test_same_numbers_as_the_app(real, app_plan, day):
    got = run_tool(real, "get_forecast", {"start_date": day, "end_date": day})["days"][0]
    want = app_plan.loc[pd.Timestamp(day)]
    assert got["forecast_guests"] == round(want.forecast)
    assert got["range_low_guests"] == round(want.low)
    assert got["range_high_guests"] == round(want.high)
    assert got["booked_so_far_guests"] == want.booked
    assert got["usual_for_weekday_guests"] == round(want.usual)
    assert got["vs_usual_pct"] == round(want.vs_usual_pct, 1)


def test_staffing_equals_pace_staffing_on_the_same_plan(real, app_plan):
    staff = P.staffing(app_plan.assign(high=app_plan.high.fillna(app_plan.forecast)).reset_index(drop=True),
                       P.DEFAULT_RULES, use="forecast").set_index("date")
    assert real.roles() == ["floor", "bar", "kitchen"]
    for day in ["2017-06-01", SAT, "2017-06-17", LAST]:
        out = run_tool(real, "get_staffing", {"date": day})
        assert out["forecast_guests"] == round(app_plan.loc[pd.Timestamp(day)].forecast)
        for s in out["staff"]:
            assert s["count"] == staff.loc[pd.Timestamp(day), s["role"].capitalize()]
    bar = run_tool(real, "get_staffing", {"date": SAT, "role": "bar"})["staff"]
    assert len(bar) == 1 and bar[0]["rule"].startswith("1 per 60 guests") and bar[0]["rule"].endswith("minimum 1")


def test_flags_follow_the_threshold(real):
    days = run_tool(real, "get_forecast", {"start_date": TODAY, "end_date": LAST})["days"]
    flagged = run_tool(real, "get_flagged_days", {"start_date": TODAY, "end_date": LAST})["days"]
    assert flagged and all(d["flag"] in ("busy", "quiet") for d in flagged)
    for d in days:
        if d["flag"] == "busy":
            assert d["vs_usual_pct"] >= 15
        elif d["flag"] == "quiet":
            assert d["vs_usual_pct"] <= -15
        else:
            assert abs(d["vs_usual_pct"]) <= 15  # 14.5-14.99 rounds to 15 but is not flagged, as in the app


def test_shapes_match_the_fake_backend(real):
    fake = FakeBackend()
    f_day = run_tool(fake, "get_forecast", {"start_date": "2017-06-01", "end_date": "2017-06-01"})["days"][0]
    r_day = run_tool(real, "get_forecast", {"start_date": TODAY, "end_date": TODAY})["days"][0]
    assert set(r_day) - {"closed"} == set(f_day) - {"closed"}
    assert r_day["closed"] is False
    assert all(isinstance(r_day[k], int) for k in r_day if k.endswith("_guests"))
    assert isinstance(r_day["vs_usual_pct"], float)  # one decimal, so it agrees with the flag
    f_st, r_st = fake.staffing(dt.date(2017, 6, 3), None), real.staffing(dt.date(2017, 6, 3), None)
    assert set(r_st) == set(f_st) and set(r_st["staff"][0]) == set(f_st["staff"][0])
    assert set(fake.accuracy()) <= set(real.accuracy())
    assert set(fake.today_and_data_range()) == set(real.today_and_data_range())


def test_accuracy_is_computed_from_the_self_check(real):
    _, check = app_run_plan(P.sample_export(), P.SAMPLE_EXPORT_DAY)
    acc = real.accuracy()
    near = check[check.days_ahead <= 7]
    assert acc["backtest_weeks"] == 8 and acc["range_percentile"] == 80
    assert acc["mean_abs_error_guests"] == round(near.pace_error.mean(), 1)
    assert acc["usual_mean_abs_error_guests"] == round(near.usual_error.mean(), 1)
    assert 0 <= acc["range_coverage_pct"] <= 100 and acc["range_coverage_days_checked"] == 28
    assert acc["mean_abs_error_guests"] < acc["usual_mean_abs_error_guests"]


def test_computed_once(real):
    real.forecast(dt.date(2017, 6, 1), dt.date(2017, 6, 2))
    cached = real._cache
    real.accuracy()
    assert real._cache is cached and real.compute_seconds["total"] > 0


def test_custom_export_with_an_earlier_export_day():
    raw = P.sample_export()
    visit = pd.to_datetime(raw["Visit date"], dayfirst=True)
    raw = raw[visit >= "2016-12-01"].reset_index(drop=True)
    rules = pd.DataFrame({"role": ["Bar", "Kitchen"], "guests_per_person": [50, 40], "minimum": [1, 1]})
    b = RealBackend(raw, export_day=pd.Timestamp("2017-04-15"), rules=rules)
    info = run_tool(b, "get_today_and_data_range", {})
    assert info["today"] == "2017-04-16" and info["dataset"] == "Reservations export of 15 Apr 2017"
    days = run_tool(b, "get_forecast", {"start_date": "2017-04-16", "end_date": "2017-05-13"})["days"]
    assert len(days) == 28 and all(not d["closed"] and d["forecast_guests"] > 0 for d in days)
    p, _ = app_run_plan(raw, pd.Timestamp("2017-04-15"))
    assert [d["forecast_guests"] for d in days] == [round(x) for x in p.forecast]
    assert b.roles() == ["bar", "kitchen"]
    st = run_tool(b, "get_staffing", {"date": "2017-04-22", "role": "kitchen"})
    want = P.staffing(p.assign(high=p.high.fillna(p.forecast)), rules).set_index("date")
    assert st["staff"] == [{"role": "kitchen", "count": int(want.loc[pd.Timestamp("2017-04-22"), "Kitchen"]),
                            "rule": "1 per 40 guests (on the forecast), rounded up, minimum 1"}]
    acc = b.accuracy()
    assert acc["range_coverage_pct"] is None and "unknown" in acc["note"]
    assert "past" in run_tool(b, "get_forecast", {"start_date": "2017-04-15", "end_date": "2017-04-20"})["error"]


def test_too_little_history_is_a_plain_error():
    b = RealBackend(P.TEMPLATE, export_day=pd.Timestamp("2026-11-13"))
    out = run_tool(b, "get_today_and_data_range", {})
    assert "error" in out and "3 weeks" in out["error"]
