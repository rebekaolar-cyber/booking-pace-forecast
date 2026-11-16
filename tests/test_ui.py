"""The app's look (app_pages/ui.py): the numbers it shows are real, and the HTML is safe."""
import numpy as np
import pandas as pd
import pytest

from app_pages import ui
from pace.data import load_hotels
from pace.models import pace_forecast, reference_days, signals


def test_the_header_example_is_real_data():
    """The header tells one story with five numbers and two curves. All of it is recomputed
    here from the benchmark data, so the page can't drift into showing made-up figures."""
    hotel = load_hotels()["City Hotel"]
    day, horizon = pd.Timestamp(ui.STORY_DAY), 7
    s = signals(hotel, day, horizon)
    assert s.booked_now == ui.STORY["booked"]
    assert round(s.booked_normally) == ui.STORY["normally"]
    assert round(s.weekday_average) == ui.STORY["weekday_average"]
    assert round(pace_forecast(s)) == ui.STORY["pace"]
    assert hotel.actual[day] == ui.STORY["actual"]

    days_before = range(60, -1, -1)
    refs = reference_days(hotel, day, horizon)
    assert [hotel.booked(day, k) for k in days_before] == ui.STORY_THIS
    normal = [np.mean([hotel.booked(r, k) for r in refs]) for k in days_before]
    assert ui.STORY_NORMAL == pytest.approx(normal, abs=0.06)  # stored to one decimal


def test_day_status_is_the_one_rule_for_colour():
    assert ui.day_status(15, 15) == "busier"
    assert ui.day_status(-15, 15) == "quieter"
    assert ui.day_status(14.9, 15) == "usual"
    assert ui.day_status(float("nan"), 15) == "usual"  # no usual number to compare with


def _day(date, **over):
    day = {"date": pd.Timestamp(date), "closed": False, "forecast": 100.0, "usual": 100.0,
           "status": "usual", "vs_usual_pct": 0.0, "sub": "", "title": ""}
    day.update(over)
    return day


def test_calendar_marks_unusual_and_closed_days():
    html = ui.calendar([
        _day("2026-11-16"),
        _day("2026-11-17", forecast=150.0, status="busier", vs_usual_pct=50.0),
        _day("2026-11-18", forecast=60.0, status="quieter", vs_usual_pct=-40.0),
        {"date": pd.Timestamp("2026-11-19"), "closed": True},
    ])
    assert html.count('class="gf-day') == 4
    assert "gf-busier" in html and "↑ 50%" in html
    assert "gf-quieter" in html and "↓ 40%" in html
    assert "Closed" in html
    # Only the two flagged days carry a percentage: an ordinary day stays quiet.
    assert html.count('class="gf-chip"') == 2


def test_calendar_bars_share_one_scale():
    """The busiest day fills its bar; the others are in proportion, so bars compare honestly."""
    html = ui.calendar([_day("2026-11-16", forecast=200.0), _day("2026-11-17", forecast=50.0)])
    assert 'style="height:100%"' in html
    assert 'style="height:25%"' in html


def test_text_from_a_file_cannot_inject_html():
    """Hover text and labels can come from an uploaded file's values, so they are escaped."""
    html = ui.calendar([_day("2026-11-16", title='"><script>alert(1)</script>', sub="<b>x</b>")])
    assert "<script>" not in html and "<b>x</b>" not in html
    assert "<script>" not in ui.stats([("<script>", "<script>", "<script>")])
    assert "<script>" not in ui.key([("bar", ui.CALM, "<script>")])


def test_error_bars_are_in_proportion():
    html = ui.error_bars([("Weekday average", 20.0, ui.BASELINE, ""), ("Booking pace", 5.0, ui.PACE, "")])
    assert "width:100.0%" in html and "width:25.0%" in html
