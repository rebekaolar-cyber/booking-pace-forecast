"""The real forecast behind the assistant's tools: wraps the booking-pace-forecast core (`pace`).

It runs the same steps, in the same order, as the Streamlit app's ``run_plan`` so the assistant and
the app show the same numbers:

    prepare -> self_check (last 8 weeks) -> build_book -> plan(28 days, range = self_check's pace_p80)

Everything is computed once, on the first tool call, and cached. "Today" is the day after the
export day: the plan's day 1 (see ``pace.models``: "Day 1 is today").
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:  # `pace` lives in the project's src/ folder
    import pace  # noqa: F401
except ImportError:
    # src/ sits next to agent/ in this repo. BPF_SRC can point elsewhere (e.g. another checkout).
    _candidates = [Path(__file__).resolve().parents[1] / "src"]
    if os.environ.get("BPF_SRC"):
        _candidates.append(Path(os.environ["BPF_SRC"]))
    for _src in _candidates:
        if (_src / "pace").is_dir():
            sys.path.insert(0, str(_src))
            break

from pace import plan as P  # noqa: E402

from agent.contract import FLAG_THRESHOLD_PCT, MAX_HORIZON_DAYS  # noqa: E402

SAMPLE_NAME = "City hotel sample (real bookings, export of 31 May 2017)"
BACKTEST_WEEKS = 8           # self_check's default, as in the app
ERROR_HORIZONS = range(1, 8)  # accuracy() averages the error over 1-7 days ahead
RANGE_PERCENTILE = 80         # pace_p80: 8 in 10 past days stayed within this error


def _int(x) -> int | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(round(x))


class RealBackend:
    """ForecastBackend on real reservations. Default: the sample export the app also uses."""

    def __init__(self, raw_export: pd.DataFrame | None = None, export_day: pd.Timestamp | None = None,
                 rules: pd.DataFrame | None = None, actual: pd.Series | None = None,
                 dayfirst: bool = True, skip_closed: bool = True, name: str | None = None):
        self._is_sample = raw_export is None
        self._raw = raw_export
        self._export_day = None if export_day is None else pd.Timestamp(export_day).normalize()
        self._rules = P.DEFAULT_RULES if rules is None else rules
        self._actual = actual
        self._dayfirst = dayfirst
        self._skip_closed = skip_closed
        self._name = name
        self._cache: dict | None = None
        self.compute_seconds: dict[str, float] = {}  # timing of the one-off computation

    # ------------------------------------------------------------------ compute once
    def _data(self) -> dict:
        if self._cache is not None:
            return self._cache
        t0 = time.perf_counter()
        raw = P.sample_export() if self._is_sample else self._raw
        prep = P.prepare(raw, P.guess_columns(raw.columns), self._dayfirst)
        res = prep.reservations
        # Same default export day as the app: the sample's 31 May 2017, else the last booking made.
        until = self._export_day or (P.SAMPLE_EXPORT_DAY if self._is_sample else res.made_on.max())
        history_days = (until - res.visit_date.min()).days
        if history_days < 21:
            raise ValueError(f"The reservations only go back {history_days} days before {until:%d %b %Y}; "
                             "the forecast needs at least 3 weeks of past visits.")
        t1 = time.perf_counter()
        # --- exactly the app's run_plan (streamlit_app.py), with 28 days
        check = P.self_check(prep, until, self._actual, skip_closed=self._skip_closed)
        t2 = time.perf_counter()
        book = P.build_book(prep, until, self._actual)
        typical = dict(zip(check.days_ahead, check.pace_p80)) if len(check) else None
        p = P.plan(book, until, days=MAX_HORIZON_DAYS, skip_closed=self._skip_closed, typical_error=typical)
        # --- the app's staffing call, with its default "Staff for: The forecast"
        staff = P.staffing(p.assign(high=p.high.fillna(p.forecast)), self._rules, use="forecast")
        t3 = time.perf_counter()
        self.compute_seconds = {"load_and_prepare": t1 - t0, "self_check": t2 - t1,
                                "plan_and_staffing": t3 - t2, "total": t3 - t0}
        self._cache = {"prep": prep, "until": until, "check": check, "plan": p.set_index("date", drop=False),
                       "staff": staff.set_index("date")}
        return self._cache

    def _row(self, day: dt.date) -> pd.Series:
        p = self._data()["plan"]
        ts = pd.Timestamp(day)
        if ts not in p.index:
            raise ValueError(f"{day} is outside the {MAX_HORIZON_DAYS}-day plan.")
        return p.loc[ts]

    # ------------------------------------------------------------------ protocol
    def today_and_data_range(self) -> dict:
        d = self._data()
        visits = d["prep"].reservations.visit_date
        until = d["until"]
        return {"today": (until + pd.Timedelta(days=1)).date().isoformat(),
                "first_booking_date": visits.min().date().isoformat(),
                "last_booking_date": visits.max().date().isoformat(),
                "dataset": self._name or (SAMPLE_NAME if self._is_sample
                                          else f"Reservations export of {until:%d %b %Y}"),
                "max_horizon_days": MAX_HORIZON_DAYS}

    def _day(self, r: pd.Series) -> dict:
        day = r.date.date()
        out = {"date": day.isoformat(), "weekday": day.strftime("%A")}
        if r.closed:
            return {**out, "closed": True}
        pct = r.vs_usual_pct
        # Flag on the unrounded percentage, like the app's "vs usual" column.
        flag = None
        if not math.isnan(pct) and abs(pct) >= FLAG_THRESHOLD_PCT:
            flag = "busy" if pct > 0 else "quiet"
        return {**out,
                "booked_so_far_guests": _int(r.booked),
                "usual_for_weekday_guests": _int(r.usual),
                "forecast_guests": _int(r.forecast),
                "range_low_guests": _int(r.low),
                "range_high_guests": _int(r.high),
                # One decimal, so the number shown always agrees with the flag (-14.5 is not
                # flagged; rounded to -15 it would look like it should be).
                "vs_usual_pct": None if math.isnan(pct) else round(float(pct), 1),
                "flag": flag,
                "closed": False}

    def forecast(self, start_date: dt.date, end_date: dt.date) -> dict:
        p = self._data()["plan"]
        sel = p[(p.date >= pd.Timestamp(start_date)) & (p.date <= pd.Timestamp(end_date))]
        return {"days": [self._day(r) for _, r in sel.iterrows()]}

    def _rule_rows(self) -> list[tuple[str, str, float, int]]:
        """(column name in the staffing table, lowercase role, guests per person, minimum),
        the rows pace.plan.staffing actually uses."""
        rows = []
        for _, r in self._rules.dropna().iterrows():
            if float(r.guests_per_person) > 0:
                rows.append((str(r.role), str(r.role).strip().lower(), float(r.guests_per_person), int(r.minimum)))
        return rows

    def roles(self) -> list[str]:
        return [role for _, role, _, _ in self._rule_rows()]

    def staffing(self, date: dt.date, role: str | None) -> dict:
        r = self._row(date)
        staff = self._data()["staff"].loc[pd.Timestamp(date)]
        out = []
        for col, name, gpp, minimum in self._rule_rows():
            if role is not None and name != role:
                continue
            rule = ("closed: no staff planned" if r.closed else
                    f"1 per {gpp:g} guests (on the forecast), rounded up, minimum {minimum}")
            out.append({"role": name, "count": int(staff[col]), "rule": rule})
        result = {"date": date.isoformat(), "forecast_guests": 0 if r.closed else _int(r.forecast), "staff": out}
        if r.closed:
            result["closed"] = True
        return result

    def accuracy(self) -> dict:
        d = self._data()
        check = d["check"]
        near = check[check.days_ahead.isin(ERROR_HORIZONS)]
        mae = round(float(near.pace_error.mean()), 1) if len(near) else None
        usual = round(float(near.usual_error.mean()), 1) if len(near) else None

        # The app's "inside the likely range" share: planned open days whose real guest number is
        # known, and how many landed between low and high.
        outcome = P.sample_outcome() if self._is_sample else self._actual
        coverage, checked = None, 0
        if outcome is not None:
            p = d["plan"]
            hit = p[~p.closed].assign(actual=p.date.map(outcome)).dropna(subset=["actual", "low", "high"])
            checked = len(hit)
            if checked:
                coverage = _int(100 * ((hit.actual >= hit.low) & (hit.actual <= hit.high)).mean())

        note = (f"Average error 1-7 days ahead, replayed on the {BACKTEST_WEEKS} weeks before the export "
                f"(weekday average alone: {usual}); ")
        note += (f"the likely range held the real number on {coverage}% of {checked} planned days."
                 if coverage is not None else
                 "range coverage is unknown: what really happened after the export is not in the data.")
        return {"backtest_weeks": BACKTEST_WEEKS, "mean_abs_error_guests": mae,
                "usual_mean_abs_error_guests": usual, "range_coverage_pct": coverage,
                "range_coverage_days_checked": checked, "range_percentile": RANGE_PERCENTILE,
                "note": note}
