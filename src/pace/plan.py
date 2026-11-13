"""Use the forecast on your own reservations: read an export, plan the next days, check accuracy.

The input is the kind of list every booking system can export: one row per reservation with the
day of the visit, the number of guests and the day it was made. A cancellation date or a status
column makes it better; a separate file with real guests per day (from the till, walk-ins
included) makes it better still.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pace.data import Hotel, load_bookings
from pace.models import pace_forecast, signals

FIELDS = {
    "visit_date": "Day of the visit",
    "guests": "Number of guests",
    "made_on": "Day the reservation was made",
    "cancelled_on": "Day it was cancelled (optional)",
    "status": "Status (optional)",
}
REQUIRED = ("visit_date", "guests", "made_on")

# Column names seen in real exports (English and Dutch). Matched after lower-casing and
# dropping spaces, dashes and underscores.
_GUESSES = {
    "visit_date": ["visitdate", "date", "datum", "reservationdate", "arrival", "arrivaldate",
                   "bookingdate", "reserveringsdatum", "dag", "day", "mealdate", "visit"],
    "guests": ["guests", "covers", "pax", "partysize", "people", "persons", "personen", "gasten",
               "aantalpersonen", "aantal", "size"],
    "made_on": ["madeon", "createdat", "created", "bookedon", "bookingmade", "dateofbooking",
                "aangemaakt", "aangemaaktop", "geboektop", "reservationmade", "bookedat", "creationdate"],
    "cancelled_on": ["cancelledon", "canceledon", "cancellationdate", "cancelledat", "canceledat",
                     "geannuleerdop", "annuleringsdatum"],
    "status": ["status", "state", "reservationstatus", "bookingstatus"],
}
_CANCELLED = re.compile(r"cancel|annul", re.I)
_NO_SHOW = re.compile(r"no.?show|niet.?versch|not.?arrived", re.I)


def _norm(name: str) -> str:
    return re.sub(r"[\s_\-./()]", "", str(name).lower())


def guess_columns(columns) -> dict[str, str | None]:
    """Best guess of which column holds which field; None where nothing fits."""
    norm = {c: _norm(c) for c in columns}
    picked: dict[str, str | None] = {}
    used = set()
    for f, names in _GUESSES.items():
        hit = next((c for n in names for c, nc in norm.items() if nc == n and c not in used), None)
        if hit is None:  # looser: the known name is part of the column name
            hit = next((c for n in names for c, nc in norm.items() if n in nc and c not in used), None)
        picked[f] = hit
        if hit:
            used.add(hit)
    return picked


def _dates(col: pd.Series, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(col, errors="coerce", dayfirst=dayfirst, format="mixed").dt.normalize()


@dataclass
class Prepared:
    reservations: pd.DataFrame          # visit_date, made_on, guests, cancelled_on, came
    dropped: int                        # rows skipped: no readable date or no guests
    notes: list[str] = field(default_factory=list)


def prepare(raw: pd.DataFrame, mapping: dict[str, str | None], dayfirst: bool = True) -> Prepared:
    """Turn an export into clean reservations. Raises ValueError with a plain message."""
    missing = [FIELDS[f] for f in REQUIRED if not mapping.get(f)]
    if missing:
        raise ValueError("Choose a column for: " + ", ".join(missing))
    df = pd.DataFrame({
        "visit_date": _dates(raw[mapping["visit_date"]], dayfirst),
        "made_on": _dates(raw[mapping["made_on"]], dayfirst),
        "guests": pd.to_numeric(raw[mapping["guests"]], errors="coerce"),
    })
    notes = []
    cancelled_on = (_dates(raw[mapping["cancelled_on"]], dayfirst) if mapping.get("cancelled_on")
                    else pd.Series(pd.NaT, index=raw.index))
    status = raw[mapping["status"]].astype(str) if mapping.get("status") else pd.Series("", index=raw.index)
    is_cancelled = status.str.contains(_CANCELLED) | cancelled_on.notna()
    no_show = status.str.contains(_NO_SHOW)
    # A cancellation without a date: we don't know when it left the book. Treat it as never
    # having been in it, which is the cautious choice.
    undated = is_cancelled & cancelled_on.isna()
    if undated.any():
        notes.append(f"{int(undated.sum())} cancellations have no cancellation date; "
                     "they are left out of the book entirely.")
    df["cancelled_on"] = cancelled_on.where(~undated, df.visit_date - pd.Timedelta(days=MAX_LEAD))
    df["came"] = ~is_cancelled & ~no_show
    ok = df.visit_date.notna() & df.made_on.notna() & df.guests.notna() & (df.guests > 0)
    df = df[ok].copy()
    df["guests"] = df.guests.round().astype(int)
    df["made_on"] = df[["made_on", "visit_date"]].min(axis=1)  # walk-ins: made on the day itself
    if no_show.any():
        notes.append(f"{int(no_show[ok].sum())} no-shows: kept in the book, not counted as guests.")
    if len(df) == 0:
        raise ValueError("No rows could be read. Check the columns and the date format.")
    return Prepared(df.reset_index(drop=True), dropped=int((~ok).sum()), notes=notes)


MAX_LEAD = 10_000


def daily_totals(raw: pd.DataFrame, date_col: str, guests_col: str, dayfirst: bool = True) -> pd.Series:
    d = pd.DataFrame({"day": _dates(raw[date_col], dayfirst),
                      "guests": pd.to_numeric(raw[guests_col], errors="coerce")}).dropna()
    return d.groupby("day").guests.sum().astype(int)


def build_book(prep: Prepared, until: pd.Timestamp, actual: pd.Series | None = None) -> Hotel:
    """The reservation book as it stood at the end of `until` (the export day).

    Anything made or cancelled after `until` is ignored, so an old export can't see the future.
    """
    r = prep.reservations
    r = r[r.made_on <= until].copy()
    r.loc[r.cancelled_on > until, "cancelled_on"] = pd.NaT
    future = r.visit_date > until
    r.loc[future, "came"] = False  # hasn't happened yet
    if actual is not None:
        actual = actual[actual.index <= until]
    last_day = max(r.visit_date.max(), until + pd.Timedelta(days=1))
    return Hotel.from_reservations("mine", r.visit_date, r.made_on, r.guests, r.cancelled_on, r.came,
                                   actual=actual, last_day=last_day)


def plan(book: Hotel, until: pd.Timestamp, days: int = 14, skip_closed: bool = True,
         typical_error: dict[int, float] | None = None) -> pd.DataFrame:
    """Forecast the `days` days after the export day, using the book as it stood that evening.

    typical_error: per days-ahead, how far off the forecast usually is (from self_check's
    pace_p80); it becomes the low-high range. Beyond the checked horizons the last one is used.
    """
    rows = []
    for h in range(1, days + 1):
        day = until + pd.Timedelta(days=h)
        s = signals(book, day, h, skip_closed=skip_closed)
        if s is None:
            rows.append({"date": day, "days_ahead": h, "closed": True})
            continue
        f = pace_forecast(s)
        err = None
        if typical_error:
            err = typical_error.get(h, typical_error[max(typical_error)] if h > max(typical_error) else None)
        rows.append({
            "date": day, "days_ahead": h, "closed": False,
            "usual": s.weekday_average, "booked": s.booked_now, "booked_normally": s.booked_normally,
            "forecast": f,
            "low": max(0.0, f - err) if err is not None else np.nan,
            "high": f + err if err is not None else np.nan,
            "vs_usual_pct": 100 * (f / s.weekday_average - 1) if s.weekday_average else np.nan,
        })
    return pd.DataFrame(rows)


def self_check(prep: Prepared, until: pd.Timestamp, actual: pd.Series | None = None, weeks: int = 8,
               horizons=range(1, 15), skip_closed: bool = True) -> pd.DataFrame:
    """Replay the last `weeks` weeks before the export day on the user's own data.

    For each past day and each horizon, rebuild the book as it stood back then (only reservations
    made and cancellations known by that morning) and compare the forecast with what happened.
    Returns per horizon: the mean absolute error of the pace forecast and of the weekday average,
    and `pace_p80`, the error that 8 out of 10 days stayed within (used as the likely range).
    """
    book = build_book(prep, until, actual)
    rows = []
    for day in pd.date_range(until - pd.Timedelta(weeks=weeks) + pd.Timedelta(days=1), until):
        if book.index(day) is None or (skip_closed and book.actual[day] == 0):
            continue
        for h in horizons:
            s = signals(book, day, h, skip_closed=skip_closed)
            if s is None:
                continue
            rows.append({"days_ahead": h, "actual": book.actual[day],
                         "pace": pace_forecast(s), "usual": s.weekday_average})
    r = pd.DataFrame(rows)
    if r.empty:
        return pd.DataFrame(columns=["days_ahead", "days_scored", "pace_error", "usual_error", "pace_p80"])
    r["pace_error"] = (r.pace - r.actual).abs()
    r["usual_error"] = (r.usual - r.actual).abs()
    return (r.groupby("days_ahead")
             .agg(days_scored=("actual", "size"), pace_error=("pace_error", "mean"),
                  usual_error=("usual_error", "mean"),
                  pace_p80=("pace_error", lambda e: float(np.quantile(e, 0.8))))
             .reset_index())


DEFAULT_RULES = pd.DataFrame({
    "role": ["Floor", "Bar", "Kitchen"],
    "guests_per_person": [25, 60, 35],
    "minimum": [2, 1, 2],
})


def staffing(p: pd.DataFrame, rules: pd.DataFrame, use: str = "forecast") -> pd.DataFrame:
    """People per role per day: guests / guests-per-person, rounded up, never below the minimum."""
    out = p[["date"]].copy()
    for _, r in rules.dropna().iterrows():
        gpp, minimum = float(r.guests_per_person), int(r.minimum)
        if gpp <= 0:
            continue
        out[str(r.role)] = [
            0 if closed else max(minimum, math.ceil(g / gpp))
            for g, closed in zip(p[use].fillna(0), p.closed)
        ]
    return out


SAMPLE_EXPORT_DAY = pd.Timestamp("2017-05-31")


def sample_export() -> pd.DataFrame:
    """The city hotel's real bookings, as an export taken on 31 May 2017 would have looked.

    Bookings made later are left out and later cancellations are not yet known, so the sample
    can't see the future. What really happened after that day is returned separately by
    `sample_outcome`.
    """
    b = load_bookings()
    b = b[(b.hotel == "City Hotel") & (b.booked_on <= SAMPLE_EXPORT_DAY)]
    cancelled = b.cancelled_on.where(b.cancelled_on <= SAMPLE_EXPORT_DAY)
    past = b.arrival <= SAMPLE_EXPORT_DAY
    status = np.where(cancelled.notna(), "Cancelled",
                      np.where(~past, "Booked",
                               np.where(b.reservation_status == "No-Show", "No-show", "Completed")))
    out = pd.DataFrame({
        "Visit date": b.arrival.dt.strftime("%d-%m-%Y"),
        "Guests": b.guests,
        "Made on": b.booked_on.dt.strftime("%d-%m-%Y"),
        "Cancelled on": cancelled.dt.strftime("%d-%m-%Y"),
        "Status": status,
    })
    # rows cancelled before the export but whose stay was in the past keep their status
    return out.sort_values("Visit date", key=lambda s: pd.to_datetime(s, dayfirst=True)).reset_index(drop=True)


def sample_outcome() -> pd.Series:
    """Guests who really arrived at the city hotel each day (to compare the sample's forecast with)."""
    b = load_bookings()
    b = b[(b.hotel == "City Hotel") & (b.reservation_status == "Check-Out")]
    return b.groupby("arrival").guests.sum()


TEMPLATE = pd.DataFrame({
    "Visit date": ["14-11-2026", "14-11-2026", "15-11-2026", "13-11-2026"],
    "Guests": [4, 2, 6, 3],
    "Made on": ["01-11-2026", "13-11-2026", "10-11-2026", "13-11-2026"],
    "Cancelled on": ["", "", "12-11-2026", ""],
    "Status": ["Booked", "Booked", "Cancelled", "Walk-in"],
})
