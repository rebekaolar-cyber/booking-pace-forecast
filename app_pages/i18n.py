"""English and Dutch for the app's own text.

Every visible sentence is written as ``tr("English", "Nederlands")`` next to the code that shows
it, so the two versions can't drift apart. The choice lives in this visitor's session only: it
never changes the page's address, so a shared link always opens the same app (in English, or in
Dutch with ``?lang=nl`` added).

Reservation data, tool results and the assistant's answers are not translated here: the
assistant already answers in the language of the question.
"""
from __future__ import annotations

import re
from typing import TypeVar

import pandas as pd
import streamlit as st

LANGUAGES = {"en": "English", "nl": "Nederlands"}
Text = TypeVar("Text")  # a string, or a list of them
STATE_KEY = "lang"  # the language switch's widget key, so the switch itself holds the choice

_DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
         "nl": ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"]}
_DAYS_SHORT = {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
               "nl": ["ma", "di", "wo", "do", "vr", "za", "zo"]}
_MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August",
                  "September", "October", "November", "December"],
           "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus",
                  "september", "oktober", "november", "december"]}
_MONTHS_SHORT = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec"],
                 "nl": ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt",
                        "nov", "dec"]}


def lang() -> str:
    """This visitor's language: 'en' or 'nl'. English outside a running app (tests, scripts)."""
    try:
        chosen = st.session_state.get(STATE_KEY)
    except Exception:  # noqa: BLE001  no Streamlit session at all
        return "en"
    return chosen if chosen in LANGUAGES else "en"


def tr(en: Text, nl: Text) -> Text:
    """The English or the Dutch version, whichever this visitor chose."""
    return nl if lang() == "nl" else en


def initial_language() -> str:
    """Language for a visitor's first page view: ?lang=nl opens in Dutch, anything else English."""
    try:
        asked = str(st.query_params.get("lang", "en")).lower()
    except Exception:  # noqa: BLE001
        return "en"
    return asked if asked in LANGUAGES else "en"


def ensure() -> None:
    """Settle this visitor's language before anything is drawn (the browser tab title too)."""
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = initial_language()


def switch() -> None:
    """The English / Nederlands switch. Draw it before any text, so the whole page follows it."""
    ensure()
    with st.container(horizontal=True, horizontal_alignment="right", key="lang_switch"):  # styled in ui.CSS
        st.segmented_control("Language / Taal", list(LANGUAGES), format_func=LANGUAGES.get,
                             key=STATE_KEY, required=True, label_visibility="collapsed")


# ---------------------------------------------------------------- dates ---------------------

def day_name(d: pd.Timestamp) -> str:
    return _DAYS[lang()][d.weekday()]


def day_short(d: pd.Timestamp) -> str:
    return _DAYS_SHORT[lang()][d.weekday()]


def month_name(d: pd.Timestamp) -> str:
    return _MONTHS[lang()][d.month - 1]


def month_short(d: pd.Timestamp) -> str:
    return _MONTHS_SHORT[lang()][d.month - 1]


def short_date(d: pd.Timestamp) -> str:
    """'Sat 3 Jun' / 'za 3 jun'."""
    return f"{day_short(d)} {d.day} {month_short(d)}"


def long_date(d: pd.Timestamp, year: bool = True) -> str:
    """'Saturday 03 June 2017' / 'zaterdag 3 juni 2017'."""
    day = f"{d.day:02d}" if lang() == "en" else str(d.day)
    return f"{day_name(d)} {day} {month_name(d)}" + (f" {d.year}" if year else "")


def month_year(d: pd.Timestamp) -> str:
    """'May 2017' / 'mei 2017'."""
    return f"{month_short(d)} {d.year}"


def vega_locale() -> dict | None:
    """Day and month names for chart axes and tooltips; None keeps the charts' English default."""
    if lang() != "nl":
        return None
    sunday_first = _DAYS["nl"][-1:] + _DAYS["nl"][:-1]
    return {"time": {
        "dateTime": "%A %e %B %Y %X", "date": "%d-%m-%Y", "time": "%H:%M:%S",
        "periods": ["AM", "PM"], "days": sunday_first,
        "shortDays": [d[:2] for d in sunday_first],
        "months": _MONTHS["nl"], "shortMonths": _MONTHS_SHORT["nl"],
    }, "number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["€ ", ""]}}


# ---------------------------------------------------------------- messages from other code --

_FIELD_NAMES_NL = {
    "Day of the visit": "Dag van het bezoek",
    "Number of guests": "Aantal gasten",
    "Day the reservation was made": "Dag waarop gereserveerd is",
    "Day it was cancelled (optional)": "Dag van annulering (optioneel)",
    "Status (optional)": "Status (optioneel)",
}

_MESSAGES_NL = [  # (English pattern, Dutch) for the plain-text notes and errors in pace/plan.py
    (r"(\d+) cancellations have no cancellation date; they are left out of the book entirely\.",
     r"\1 annuleringen hebben geen annuleringsdatum; die tellen helemaal niet mee."),
    (r"(\d+) no-shows: kept in the book, not counted as guests\.",
     r"\1 no-shows: blijven in het boek, maar tellen niet als gasten."),
    (r"No rows could be read\. Check the columns and the date format\.",
     "Er kon geen enkele rij worden gelezen. Controleer de kolommen en de datumnotatie."),
    (r"A walk-in share must be at least 0% and below 100%\.",
     "Het aandeel walk-ins moet minstens 0% en minder dan 100% zijn."),
]


def field_name(english: str) -> str:
    return _FIELD_NAMES_NL.get(english, english) if lang() == "nl" else english


def message(english: str) -> str:
    """A note or error from the forecast code, in this visitor's language."""
    if lang() != "nl":
        return english
    if english.startswith("Choose a column for: "):
        names = english.removeprefix("Choose a column for: ").split(", ")
        return "Kies een kolom voor: " + ", ".join(field_name(n) for n in names)
    for pattern, dutch in _MESSAGES_NL:
        if re.fullmatch(pattern, english):
            return re.sub(pattern, dutch, english)
    return english
