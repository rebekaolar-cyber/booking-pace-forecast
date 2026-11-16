"""Deterministic stand-in for the forecast, used by tests and offline development.

Shaped like the real backend (today = Thursday 2017-06-01, the morning after the 31 May sample
export) but every number is
made up, so nothing here may ever be shown to a real user as a forecast.
"""
from __future__ import annotations

import datetime as dt

from agent.contract import FLAG_THRESHOLD_PCT, MAX_HORIZON_DAYS

class FakeBackend:
    """Deterministic backend for tests and offline development. Shaped like the real one:
    today = Thursday 2017-06-01 (the morning after the 31 May sample export), staffing rules per role."""

    TODAY = dt.date(2017, 6, 1)
    _USUAL = {0: 60, 1: 55, 2: 58, 3: 70, 4: 95, 5: 118, 6: 80}  # Mon..Sun
    _RULES = {"bar": 40, "kitchen": 30, "floor": 25}  # guests per staff member

    def today_and_data_range(self) -> dict:
        return {"today": self.TODAY.isoformat(), "first_booking_date": "2015-07-01",
                "last_booking_date": "2017-08-31", "dataset": "Fake city hotel (tests)",
                "max_horizon_days": MAX_HORIZON_DAYS}

    def _day(self, d: dt.date) -> dict:
        usual = self._USUAL[d.weekday()]
        offset = ((d.toordinal() * 7) % 11) - 5  # -5..5, deterministic
        bump = 1.25 if d.day == 10 else (0.8 if d.day == 20 else 1.0)
        fc = round((usual + offset) * bump)
        pct = round((fc - usual) / usual * 100)
        flag = "busy" if pct >= FLAG_THRESHOLD_PCT else ("quiet" if pct <= -FLAG_THRESHOLD_PCT else None)
        lead = (d - self.TODAY).days
        return {"date": d.isoformat(), "weekday": d.strftime("%A"),
                "booked_so_far_guests": max(0, fc - 2 * lead - 5),
                "usual_for_weekday_guests": usual, "forecast_guests": fc,
                "range_low_guests": fc - 9 - lead // 3, "range_high_guests": fc + 9 + lead // 3,
                "vs_usual_pct": pct, "flag": flag, "closed": False}

    def forecast(self, start_date: dt.date, end_date: dt.date) -> dict:
        n = (end_date - start_date).days + 1
        return {"days": [self._day(start_date + dt.timedelta(days=i)) for i in range(n)]}

    def roles(self) -> list[str]:
        return list(self._RULES)

    def staffing(self, date: dt.date, role: str | None) -> dict:
        fc = self._day(date)["forecast_guests"]
        roles = [role] if role else list(self._RULES)
        staff = [{"role": r, "count": max(1, -(-fc // self._RULES[r])),
                  "rule": f"1 per {self._RULES[r]} guests, rounded up, minimum 1"} for r in roles]
        return {"date": date.isoformat(), "forecast_guests": fc, "staff": staff}

    def accuracy(self) -> dict:
        return {"backtest_weeks": 8, "mean_abs_error_guests": 8.6,
                "range_coverage_pct": 80, "range_percentile": 80}
