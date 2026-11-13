"""Load the hotel bookings and rebuild the reservation book.

The dataset has one row per booking. From it we rebuild two things for every arrival day:

* ``actual``: the guests who really arrived (adults + children + babies of stays that
  checked out; cancellations and no-shows don't count).
* ``book``: how many guests were in the reservation book for that day, k days before it.
  A booking is in the book from the day after it was made until the day it was cancelled.

Everything a forecast uses must be known *before* the day it is made. Keeping the book as a
day x days-before table makes that rule easy to follow: a forecast made k days ahead may only
look at column k or further left in time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "hotels.csv.gz"
MAX_DAYS_BEFORE = 400  # longer than the longest lead time we use

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


def load_bookings(path: str | Path = DATA_FILE) -> pd.DataFrame:
    """One row per booking, with the dates the rest of the code needs."""
    df = pd.read_csv(path)
    df["arrival"] = pd.to_datetime(pd.DataFrame({
        "year": df.arrival_date_year,
        "month": df.arrival_date_month.map(_MONTHS),
        "day": df.arrival_date_day_of_month,
    }))
    df["booked_on"] = df.arrival - pd.to_timedelta(df.lead_time, unit="D")
    df["guests"] = df[["adults", "children", "babies"]].fillna(0).sum(axis=1).astype(int)
    status_date = pd.to_datetime(df.reservation_status_date)
    df["cancelled_on"] = status_date.where(df.reservation_status == "Canceled")
    return df[["hotel", "arrival", "booked_on", "lead_time", "guests",
               "reservation_status", "cancelled_on"]]


@dataclass
class Hotel:
    """One hotel's history: guests per day and the reservation book per day."""

    name: str
    days: pd.DatetimeIndex
    actual: pd.Series          # guests who arrived, per day
    book: np.ndarray           # book[i, k] = guests in the book for days[i], k days before it

    @classmethod
    def from_bookings(cls, name: str, bookings: pd.DataFrame) -> "Hotel":
        """From the hotel dataset's columns (see load_bookings)."""
        b = bookings[bookings.hotel == name]
        return cls.from_reservations(
            name,
            arrival=b.arrival,
            made_on=b.booked_on,
            guests=b.guests,
            cancelled_on=b.cancelled_on,
            came=b.reservation_status == "Check-Out",
        )

    @classmethod
    def from_reservations(cls, name: str, arrival: pd.Series, made_on: pd.Series, guests: pd.Series,
                          cancelled_on: pd.Series, came: pd.Series,
                          actual: pd.Series | None = None, last_day: pd.Timestamp | None = None) -> "Hotel":
        """From any reservation list.

        arrival, made_on, cancelled_on: dates (cancelled_on empty if not cancelled).
        came: True for reservations that turned into guests (not cancelled, not a no-show).
        actual: guests per day, if known separately (for example from the till, walk-ins
                included). Without it, guests per day = the reservations that came.
        """
        arrival = pd.to_datetime(arrival).dt.normalize()
        made_on = pd.to_datetime(made_on).dt.normalize()
        cancelled_on = pd.to_datetime(cancelled_on).dt.normalize()
        guests = guests.astype(int)
        first = arrival.min() if actual is None else min(arrival.min(), actual.index.min())
        last = max(arrival.max(), last_day) if last_day is not None else arrival.max()
        days = pd.date_range(first, last, freq="D")
        if actual is None:
            actual = guests[came.to_numpy()].groupby(arrival[came.to_numpy()]).sum()
        actual = actual.reindex(days, fill_value=0).astype(int)

        # A reservation for day d made `lead` days before it is in the book at the start of every
        # day d - k with k < lead (it was made the day before), and it stays there while the
        # cancellation date is on or after d - k, i.e. k >= d - cancelled_on.
        row = (arrival - days[0]).dt.days.to_numpy()
        lead = (arrival - made_on).dt.days.clip(lower=0).to_numpy()
        hi = np.minimum(lead, MAX_DAYS_BEFORE)
        lo = np.zeros(len(arrival), dtype=int)
        cancelled = cancelled_on.notna().to_numpy()
        gap = (arrival - cancelled_on).dt.days.to_numpy()
        lo[cancelled] = np.clip(gap[cancelled], 0, None)
        keep = hi > lo
        diff = np.zeros((len(days), MAX_DAYS_BEFORE + 1))
        g = guests.to_numpy()
        np.add.at(diff, (row[keep], lo[keep]), g[keep])
        np.add.at(diff, (row[keep], hi[keep]), -g[keep])
        book = np.cumsum(diff, axis=1)[:, :MAX_DAYS_BEFORE].round().astype(int)
        return cls(name=name, days=days, actual=actual, book=book)

    def index(self, day: pd.Timestamp) -> int | None:
        i = (day - self.days[0]).days
        return i if 0 <= i < len(self.days) else None

    def booked(self, day: pd.Timestamp, days_before: int) -> int:
        """Guests in the book for `day` at the start of the day `days_before` days earlier."""
        i = self.index(day)
        if i is None or days_before >= MAX_DAYS_BEFORE:
            return 0
        return int(self.book[i, days_before])


def load_hotels(path: str | Path = DATA_FILE) -> dict[str, Hotel]:
    bookings = load_bookings(path)
    return {name: Hotel.from_bookings(name, bookings) for name in sorted(bookings.hotel.unique())}
