"""The forecasts, from simplest to smartest.

All of them answer the same question: standing at the start of a day, how many guests will
arrive on a day `horizon` days from now? Day 1 is today, day 7 is six days from now.

* ``weekday_average``  - the last 6 same weekdays, averaged. What most managers do by feel.
* ``same_day_last_year`` - what happened 52 weeks ago.
* ``pace``              - the weekday average, corrected by how full the book is compared with
                          what is normal this many days ahead. Simple enough to check by hand.
* ``gradient_boosting`` - a machine-learning model trained on the same signals (see ml.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pace.data import Hotel

WEEKS = 6              # same weekdays averaged
MIN_WEEKS = 3          # fewer than this: no forecast
MAX_CORRECTION = 0.5   # the book may move the forecast by at most half the average


@dataclass(frozen=True)
class Signals:
    """Everything known at forecast time that the models use."""

    day: pd.Timestamp
    horizon: int
    weekday_average: float
    booked_now: int
    booked_normally: float
    same_day_last_year: float


def reference_days(hotel: Hotel, day: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    """The most recent same weekdays that were already over when the forecast was made."""
    run_day = day - pd.Timedelta(days=horizon - 1)
    found = []
    for k in range(1, WEEKS + 9):
        ref = day - pd.Timedelta(weeks=k)
        if ref < run_day and hotel.index(ref) is not None:
            found.append(ref)
        if len(found) == WEEKS:
            break
    return found


def signals(hotel: Hotel, day: pd.Timestamp, horizon: int) -> Signals | None:
    refs = reference_days(hotel, day, horizon)
    if len(refs) < MIN_WEEKS:
        return None
    k = horizon - 1  # days before the arrival day
    last_year = day - pd.Timedelta(weeks=52)
    return Signals(
        day=day,
        horizon=horizon,
        weekday_average=float(np.mean([hotel.actual[r] for r in refs])),
        booked_now=hotel.booked(day, k),
        # Each earlier day's book, the same number of days before it.
        booked_normally=float(np.mean([hotel.booked(r, k) for r in refs])),
        same_day_last_year=(float(hotel.actual[last_year])
                            if hotel.index(last_year) is not None else np.nan),
    )


def pace_forecast(s: Signals) -> float:
    correction = s.booked_now - s.booked_normally
    limit = MAX_CORRECTION * s.weekday_average
    return max(0.0, s.weekday_average + max(-limit, min(limit, correction)))
