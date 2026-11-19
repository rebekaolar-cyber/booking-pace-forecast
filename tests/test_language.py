"""The English / Nederlands switch: the whole app works in both, and the link never changes."""
from pathlib import Path

import pandas as pd
import pytest

from app_pages import i18n
from pace import plan as P

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return AppTest.from_file(str(APP), default_timeout=300)


def _labels(at):
    return [t.label for t in at.tabs]


def test_the_app_opens_in_english_and_switches_to_dutch_without_touching_the_link(app):
    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Plan the next two weeks" in _labels(at)

    at.segmented_control(key=i18n.STATE_KEY).set_value("nl").run()
    assert not at.exception, [e.value for e in at.exception]
    assert _labels(at) == ["Plan de komende twee weken", "Hoe nauwkeurig is het?",
                           "Vraag het de assistent"]
    assert any("API key" in i.value and "assistent" in i.value for i in at.info)
    assert "lang" not in at.query_params  # the switch never rewrites the shared address

    at.segmented_control(key=i18n.STATE_KEY).set_value("en").run()
    assert not at.exception
    assert "Plan the next two weeks" in _labels(at)


def test_settings_survive_a_language_switch(app):
    at = app.run()
    at.segmented_control(key="proof_hotel").set_value("Resort Hotel").run()
    at.slider(key="proof_horizon").set_value(14).run()
    at.segmented_control(key=i18n.STATE_KEY).set_value("nl").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.segmented_control(key="proof_hotel").value == "Resort Hotel"
    assert at.slider(key="proof_horizon").value == 14


def test_a_dutch_link_opens_in_dutch(app):
    app.query_params["lang"] = "nl"
    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _labels(at)[0] == "Plan de komende twee weken"


def test_own_reservations_page_works_in_dutch(app):
    at = app.run()
    at.segmented_control(key=i18n.STATE_KEY).set_value("nl").run()
    at.segmented_control(key="source").set_value("own").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Je reserveringen" in u.label for u in at.file_uploader)


def test_dates_read_naturally_in_both_languages():
    day = pd.Timestamp("2017-06-03")  # a Saturday
    assert i18n.short_date(day) == "Sat 3 Jun"  # outside a running app the language is English
    assert i18n.long_date(day) == "Saturday 03 June 2017"
    assert i18n.vega_locale() is None


def test_every_message_from_the_forecast_code_has_a_dutch_version(monkeypatch):
    monkeypatch.setattr(i18n, "lang", lambda: "nl")
    assert i18n.short_date(pd.Timestamp("2017-06-03")) == "za 3 jun"
    assert i18n.long_date(pd.Timestamp("2017-06-03"), year=False) == "zaterdag 3 juni"
    for english in P.FIELDS.values():
        assert i18n.field_name(english) != english
    assert i18n.message("Choose a column for: Day of the visit, Number of guests") == \
        "Kies een kolom voor: Dag van het bezoek, Aantal gasten"
    raw = pd.DataFrame({"Visit date": ["01-06-2017", "02-06-2017"], "Guests": [2, 3],
                        "Made on": ["01-05-2017", "01-05-2017"], "Status": ["Cancelled", "No-show"]})
    prep = P.prepare(raw, P.guess_columns(raw.columns), dayfirst=True)
    assert prep.notes  # one undated cancellation, one no-show
    for note in prep.notes:
        assert i18n.message(note) != note, note
