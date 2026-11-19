"""The assistant on a visitor's own upload: same numbers as the Plan tab, one conversation per file."""
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
P = pytest.importorskip("pace.plan")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from agent.contract import run_tool  # noqa: E402
from agent.real_backend import RealBackend  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
SAT = "2017-06-03"


@pytest.fixture(scope="module")
def export():
    """The sample hotel's export, renamed to Dutch headers as a restaurant's system might write them."""
    return P.sample_export().rename(columns={"Visit date": "Datum bezoek", "Guests": "Aantal personen",
                                             "Made on": "Geboekt op", "Cancelled on": "Geannuleerd op"})


def test_walk_ins_reach_the_assistant_exactly_as_on_the_plan_tab(export):
    mapping = P.guess_columns(export.columns)
    assert mapping["visit_date"] == "Datum bezoek"  # Dutch headers are recognised
    plain = RealBackend(raw_export=export, export_day=P.SAMPLE_EXPORT_DAY, mapping=mapping)
    walk = RealBackend(raw_export=export, export_day=P.SAMPLE_EXPORT_DAY, mapping=mapping, walk_in=(0.4, 0.2))

    day_plain = run_tool(plain, "get_forecast", {"start_date": SAT, "end_date": SAT})["days"][0]
    day_walk = run_tool(walk, "get_forecast", {"start_date": SAT, "end_date": SAT})["days"][0]

    # Saturday: 40% of all guests walk in, so reservations are 60% of the day.
    expected = P.add_walk_ins(pd.DataFrame({"date": [pd.Timestamp(SAT)], "closed": [False],
                                            "forecast": [float(day_plain["forecast_guests"])]}), 0.4, 0.2)
    assert day_walk["forecast_guests"] == pytest.approx(expected.forecast[0], abs=1)
    assert day_walk["from_reservations_guests"] == pytest.approx(day_plain["forecast_guests"], abs=1)
    assert day_walk["walk_ins_estimate_guests"] == pytest.approx(
        day_walk["forecast_guests"] - day_walk["from_reservations_guests"], abs=1)
    assert day_walk["booked_so_far_guests"] == day_plain["booked_so_far_guests"]  # the real book
    assert "walk_ins_estimate_guests" not in day_plain


def test_the_visitors_column_choice_is_used_not_guessed_again(export):
    odd = export.rename(columns={"Datum bezoek": "kolom A"})
    mapping = {**P.guess_columns(export.columns), "visit_date": "kolom A"}
    backend = RealBackend(raw_export=odd, export_day=P.SAMPLE_EXPORT_DAY, mapping=mapping)
    assert run_tool(backend, "get_today_and_data_range", {})["today"] == "2017-06-01"


def test_an_upload_on_the_plan_tab_reaches_the_assistant_tab(export, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    recent = export[pd.to_datetime(export["Datum bezoek"], dayfirst=True) >= "2017-01-01"]
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at.segmented_control(key="source").set_value("own").run()
    at.file_uploader(key="upload").upload("reserveringen.csv", recent.to_csv(index=False).encode(),
                                          "text/csv").run()
    assert not at.exception, [e.value for e in at.exception]

    ask = at.segmented_control(key="ask_about")
    assert ask.value == "own"  # a fresh upload is what the visitor wants to ask about
    assert any("your own reservations" in i.value for i in at.info)
    assert any("today is Thursday 01 June 2017" in i.value for i in at.info)

    ask.set_value("sample").run()
    assert not at.exception
    assert any("sample hotel" in i.value for i in at.info)

    # Back to the sample data on the Plan tab: the assistant forgets the upload.
    at.segmented_control(key="source").set_value("sample").run()
    assert not at.exception
    assert not _has(at, "ask_about")


def _has(at, key):
    try:
        at.segmented_control(key=key)
        return True
    except KeyError:
        return False


def test_staffing_for_the_top_of_the_range_matches_the_plan_tab(export):
    mapping = P.guess_columns(export.columns)
    high = RealBackend(raw_export=export, export_day=P.SAMPLE_EXPORT_DAY, mapping=mapping, staff_use="high")
    plain = RealBackend(raw_export=export, export_day=P.SAMPLE_EXPORT_DAY, mapping=mapping)
    day = high._data()["plan"].loc[pd.Timestamp(SAT)]
    expected = P.staffing(pd.DataFrame([day]).assign(high=[day.high]), P.DEFAULT_RULES, use="high")
    got = run_tool(high, "get_staffing", {"date": SAT})
    assert {s["role"]: s["count"] for s in got["staff"]} == {
        c.lower(): int(expected[c].iloc[0]) for c in expected.columns[1:]}
    assert got["staffed_for_guests"] == round(day.high)
    assert "top of the likely range" in got["staff"][0]["rule"]
    assert "staffed_for_guests" not in run_tool(plain, "get_staffing", {"date": SAT})


def test_a_failed_upload_does_not_leave_the_old_file_with_the_assistant(export, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    recent = export[pd.to_datetime(export["Datum bezoek"], dayfirst=True) >= "2017-01-01"]
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at.segmented_control(key="source").set_value("own").run()
    at.file_uploader(key="upload").upload("goed.csv", recent.to_csv(index=False).encode(), "text/csv").run()
    assert _has(at, "ask_about")
    too_short = export[pd.to_datetime(export["Datum bezoek"], dayfirst=True) >= "2017-05-25"]
    at.file_uploader(key="upload").upload("kort.csv", too_short.to_csv(index=False).encode(), "text/csv").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.error  # the Plan tab explains what's wrong with the file
    assert not _has(at, "ask_about")  # and the assistant is back on the sample hotel


def test_the_upload_leaves_the_session_when_the_visitor_goes_back_to_the_sample(export, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    recent = export[pd.to_datetime(export["Datum bezoek"], dayfirst=True) >= "2017-01-01"]
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at.segmented_control(key="source").set_value("own").run()
    at.file_uploader(key="upload").upload("goed.csv", recent.to_csv(index=False).encode(), "text/csv").run()
    assert "own_backend" in at.session_state
    at.segmented_control(key="source").set_value("sample").run()
    assert "own_backend" not in at.session_state
