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
        b = bookings[bookings.hotel == name]
        days = pd.date_range(b.arrival.min(), b.arrival.max(), freq="D")
        arrived = b[b.reservation_status == "Check-Out"].groupby("arrival").guests.sum()
        actual = arrived.reindex(days, fill_value=0).astype(int)

        # A booking for day d made `lead` days before it is in the book at the start of every
        # day d - k with k < lead (it was made the day before), and it stays there while the
        # cancellation date is on or after d - k, i.e. k >= d - cancelled_on.
        row = (b.arrival - days[0]).dt.days.to_numpy()
        hi = np.minimum(b.lead_time.to_numpy(), MAX_DAYS_BEFORE)
        lo = np.zeros(len(b), dtype=int)
        cancelled = b.cancelled_on.notna().to_numpy()
        gap = (b.arrival - b.cancelled_on).dt.days.to_numpy()
        lo[cancelled] = np.clip(gap[cancelled], 0, None)
        keep = hi > lo
        diff = np.zeros((len(days), MAX_DAYS_BEFORE + 1))
        g = b.guests.to_numpy()
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
