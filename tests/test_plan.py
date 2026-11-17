"""Using the forecast on your own export: reading it, never seeing the future, staffing."""
import pandas as pd
import pytest

from pace import plan as P

D = pd.Timestamp


def weekly_export(weeks=12, guests=20, extra_on_last=0, start="2026-08-07"):
    """One reservation per Friday for `guests`, made 10 days ahead; the last Friday is in the future."""
    rows = []
    for w in range(weeks):
        day = D(start) + pd.Timedelta(weeks=w)
        g = guests + (extra_on_last if w == weeks - 1 else 0)
        rows.append({"Datum": f"{day:%d-%m-%Y} 19:00", "Personen": g,
                     "Aangemaakt op": f"{day - pd.Timedelta(days=10):%d-%m-%Y} 11:30",
                     "Status": "Gezeten", "Naam": "x"})
    return pd.DataFrame(rows)


def test_dutch_column_names_are_recognised():
    m = P.guess_columns(["Datum", "Naam", "Aantal personen", "Aangemaakt op", "Status", "Geannuleerd op"])
    assert m == {"visit_date": "Datum", "guests": "Aantal personen", "made_on": "Aangemaakt op",
                 "cancelled_on": "Geannuleerd op", "status": "Status"}


def test_english_export_column_names_are_recognised():
    m = P.guess_columns(["Reservation date", "Covers", "Created at", "Cancelled at", "Booking status", "Email"])
    assert m["visit_date"] == "Reservation date" and m["guests"] == "Covers"
    assert m["made_on"] == "Created at" and m["cancelled_on"] == "Cancelled at"
    assert m["status"] == "Booking status"


def test_missing_required_column_gives_a_plain_message():
    with pytest.raises(ValueError, match="Number of guests"):
        P.prepare(pd.DataFrame({"a": [1]}), {"visit_date": "a", "guests": None, "made_on": "a"})


def test_statuses_and_day_first_dates():
    raw = pd.DataFrame({
        "date": ["03-02-2026", "03-02-2026", "03-02-2026", "03-02-2026"],
        "guests": [2, 4, 3, 5],
        "made": ["01-02-2026", "01-02-2026", "01-02-2026", "03-02-2026"],
        "status": ["Gezeten", "Geannuleerd", "No-show", "Walk-in"],
    })
    prep = P.prepare(raw, {"visit_date": "date", "guests": "guests", "made_on": "made", "status": "status"})
    r = prep.reservations
    assert (r.visit_date == D("2026-02-03")).all()          # 3 February, not 2 March
    assert r.came.tolist() == [True, False, False, True]
    assert any("no cancellation date" in n for n in prep.notes)


def test_made_after_the_visit_is_treated_as_a_walk_in():
    raw = pd.DataFrame({"d": ["2026-02-03"], "g": [2], "m": ["2026-02-05"]})
    r = P.prepare(raw, {"visit_date": "d", "guests": "g", "made_on": "m"}, dayfirst=False).reservations
    assert r.made_on.iloc[0] == D("2026-02-03")


def test_plan_ignores_bookings_made_after_the_export_day():
    """An old export date must give the same plan whether or not later bookings are in the file."""
    early = weekly_export()
    later = pd.concat([early, pd.DataFrame([{"Datum": "30-10-2026 19:00", "Personen": 50,
                                             "Aangemaakt op": "29-10-2026 09:00", "Status": "Bevestigd",
                                             "Naam": "x"}])])
    until = D("2026-10-25")
    plans = []
    for raw in (early, later):
        prep = P.prepare(raw, P.guess_columns(raw.columns))
        plans.append(P.plan(P.build_book(prep, until), until, days=7))
    pd.testing.assert_frame_equal(plans[0], plans[1])


def test_plan_reads_the_book_ahead():
    raw = weekly_export(extra_on_last=10)          # the coming Friday has 10 more booked than usual
    prep = P.prepare(raw, P.guess_columns(raw.columns))
    until = D("2026-10-17")                         # a Saturday; the last Friday (23 Oct) is 6 days later
    p = P.plan(P.build_book(prep, until), until, days=7)
    friday = p[p.date == D("2026-10-23")].iloc[0]
    assert friday.usual == 20 and friday.booked == 30 and friday.forecast == 30
    assert p[p.date != D("2026-10-23")].closed.all()   # every other weekday has never had guests


def test_self_check_rebuilds_the_past_book():
    raw = weekly_export(weeks=16)
    prep = P.prepare(raw, P.guess_columns(raw.columns))
    until = D("2026-11-20")
    chk = P.self_check(prep, until, horizons=[1, 7])
    assert set(chk.days_ahead) == {1, 7}
    assert (chk.pace_error == 0).all()              # a perfectly regular place is forecast perfectly


def _plan_for(days):
    """A small plan: Thu 19, Fri 20, Sat 21, Sun 22, Mon 23 Nov 2026, the Monday closed."""
    dates = pd.date_range("2026-11-19", periods=days)
    return pd.DataFrame({"date": dates, "closed": [d.weekday() == 0 for d in dates],
                         "forecast": 60.0, "low": 48.0, "high": 72.0, "usual": 54.0, "booked": 30})


def test_walk_ins_are_a_share_of_all_guests_40_on_fri_to_sun_20_on_weekdays():
    p = P.add_walk_ins(_plan_for(5))
    p.index = p.date.dt.day_name()
    assert p.loc["Thursday", "forecast"] == pytest.approx(75)        # 60 / 0.8
    assert p.loc["Thursday", "walk_ins"] == pytest.approx(15)
    for day in ("Friday", "Saturday", "Sunday"):
        assert p.loc[day, "forecast"] == pytest.approx(100)          # 60 / 0.6
        assert p.loc[day, "walk_ins"] == pytest.approx(40)
        assert p.loc[day, "reserved"] == 60 and p.loc[day, "booked"] == 30   # the book is untouched
        assert p.loc[day, "low"] == pytest.approx(80) and p.loc[day, "high"] == pytest.approx(120)
    assert p.loc["Monday", "forecast"] == 60 and pd.isna(p.loc["Monday", "walk_in_share"])  # closed


def test_walk_ins_keep_the_vs_usual_comparison_and_take_custom_shares():
    p = _plan_for(2)
    p["vs_usual_pct"] = 100 * (p.forecast / p.usual - 1)
    out = P.add_walk_ins(p, weekend_share=0.5, weekday_share=0.0)
    assert out.forecast.tolist() == pytest.approx([60, 120])         # Thu unchanged, Fri doubled
    assert (out.forecast / out.usual).tolist() == pytest.approx((p.forecast / p.usual).tolist())
    with pytest.raises(ValueError):
        P.add_walk_ins(p, weekend_share=1.0)


def test_walk_in_rows_in_the_export_are_detected():
    raw = pd.DataFrame({"Status": ["Booked", "Walk-in"]})
    assert P.has_walk_in_rows(raw, {"status": "Status"})
    assert not P.has_walk_in_rows(raw.iloc[:1], {"status": "Status"})
    assert not P.has_walk_in_rows(raw, {"status": None})


def test_staffing_rounds_up_and_respects_the_minimum():
    p = pd.DataFrame({"date": [D("2026-01-01"), D("2026-01-02"), D("2026-01-03")],
                      "forecast": [10.0, 51.0, 0.0], "closed": [False, False, True]})
    rules = pd.DataFrame({"role": ["Floor"], "guests_per_person": [25], "minimum": [2]})
    assert P.staffing(p, rules).Floor.tolist() == [2, 3, 0]


def test_sample_export_cannot_see_past_its_export_day():
    s = P.sample_export()
    made = pd.to_datetime(s["Made on"], dayfirst=True)
    cancelled = pd.to_datetime(s["Cancelled on"], dayfirst=True)
    assert made.max() <= P.SAMPLE_EXPORT_DAY
    assert cancelled.max() <= P.SAMPLE_EXPORT_DAY
    future = pd.to_datetime(s["Visit date"], dayfirst=True) > P.SAMPLE_EXPORT_DAY
    assert set(s.Status[future]) <= {"Booked", "Cancelled"}
