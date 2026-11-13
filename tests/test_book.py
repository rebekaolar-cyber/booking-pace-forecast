"""The reservation book is rebuilt correctly, and no forecast can see the future."""
import pandas as pd
import pytest

from pace.data import Hotel
from pace.models import pace_forecast, reference_days, signals

D = pd.Timestamp


def bookings(rows):
    """rows: (arrival, lead_time, guests, status, cancelled_on or None)"""
    df = pd.DataFrame(rows, columns=["arrival", "lead_time", "guests", "reservation_status", "cancelled_on"])
    df["hotel"] = "H"
    df["arrival"] = pd.to_datetime(df.arrival)
    df["cancelled_on"] = pd.to_datetime(df.cancelled_on)
    df["booked_on"] = df.arrival - pd.to_timedelta(df.lead_time, unit="D")
    return df


def test_booking_enters_the_book_the_day_after_it_is_made():
    h = Hotel.from_bookings("H", bookings([("2017-03-10", 5, 2, "Check-Out", None)]))
    assert h.booked(D("2017-03-10"), 4) == 2      # start of 6 Mar: made on 5 Mar, so in
    assert h.booked(D("2017-03-10"), 5) == 0      # start of 5 Mar: not made yet
    assert h.booked(D("2017-03-10"), 0) == 2


def test_cancelled_booking_leaves_the_book_after_its_cancellation_day():
    h = Hotel.from_bookings("H", bookings([("2017-03-10", 10, 3, "Canceled", "2017-03-07")]))
    assert h.booked(D("2017-03-10"), 3) == 3      # start of 7 Mar: still in
    assert h.booked(D("2017-03-10"), 2) == 0      # start of 8 Mar: gone
    assert h.actual[D("2017-03-10")] == 0         # cancellations never count as guests


def test_same_day_booking_is_never_in_the_book_but_counts_as_a_guest():
    h = Hotel.from_bookings("H", bookings([("2017-03-10", 0, 4, "Check-Out", None)]))
    assert h.booked(D("2017-03-10"), 0) == 0
    assert h.actual[D("2017-03-10")] == 4


def test_no_show_stays_in_the_book_but_is_not_a_guest():
    h = Hotel.from_bookings("H", bookings([("2017-03-10", 3, 2, "No-Show", None)]))
    assert h.booked(D("2017-03-10"), 0) == 2
    assert h.actual[D("2017-03-10")] == 0


def weekly_hotel(guests_per_day=10, extra_booked=0):
    """Six weeks of Fridays with `guests_per_day` booked 7 days ahead, then one more Friday."""
    rows = [(str((D("2017-01-06") + pd.Timedelta(weeks=w)).date()), 7, guests_per_day, "Check-Out", None)
            for w in range(6)]
    target = D("2017-01-06") + pd.Timedelta(weeks=6)
    rows.append((str(target.date()), 7, guests_per_day + extra_booked, "Check-Out", None))
    return Hotel.from_bookings("H", bookings(rows)), target


def test_reference_days_are_all_over_before_the_forecast_is_made():
    h, target = weekly_hotel()
    for horizon in (1, 7, 8, 14):
        run_day = target - pd.Timedelta(days=horizon - 1)
        assert all(r < run_day for r in reference_days(h, target, horizon))


def test_pace_moves_the_forecast_by_the_extra_bookings():
    h, target = weekly_hotel(guests_per_day=10, extra_booked=4)
    s = signals(h, target, horizon=3)
    assert s.weekday_average == 10
    assert s.booked_now == 14 and s.booked_normally == 10
    assert pace_forecast(s) == 14


def test_pace_correction_is_capped_at_half_the_average():
    h, target = weekly_hotel(guests_per_day=10, extra_booked=50)
    assert pace_forecast(signals(h, target, horizon=3)) == 15


def test_no_forecast_without_three_earlier_weeks():
    h = Hotel.from_bookings("H", bookings([("2017-03-10", 3, 2, "Check-Out", None)]))
    assert signals(h, D("2017-03-10"), 1) is None


@pytest.mark.parametrize("horizon", [1, 7, 30])
def test_signals_ignore_bookings_made_after_the_forecast_day(horizon):
    """Adding a booking made on the forecast day itself must not change that forecast."""
    fridays = [(str((D("2017-01-06") + pd.Timedelta(weeks=w)).date()), 7, 10, "Check-Out", None)
               for w in range(12)]
    target = D("2017-01-06") + pd.Timedelta(weeks=11)
    # lead time horizon-1 means it was made on the run day, so it isn't in that morning's book
    late = (str(target.date()), horizon - 1, 99, "Check-Out", None)
    without = signals(Hotel.from_bookings("H", bookings(fridays)), target, horizon)
    with_late = signals(Hotel.from_bookings("H", bookings(fridays + [late])), target, horizon)
    assert without is not None
    for field in ("weekday_average", "booked_now", "booked_normally"):
        assert getattr(with_late, field) == getattr(without, field)
