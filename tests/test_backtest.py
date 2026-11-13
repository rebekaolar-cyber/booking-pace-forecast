"""The real dataset loads, the split is clean, and the saved results are what the code produces."""
import json
from pathlib import Path

import pandas as pd
import pytest

from pace.backtest import TEST, TRAIN, build_rows, run
from pace.data import load_bookings, load_hotels

REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.fixture(scope="module")
def hotels():
    return load_hotels()


def test_dataset_is_complete():
    b = load_bookings()
    assert len(b) == 119_390
    assert set(b.hotel) == {"City Hotel", "Resort Hotel"}


def test_train_and_test_periods_do_not_overlap():
    assert TRAIN[1] < TEST[0]


def test_book_never_exceeds_what_was_ever_booked(hotels):
    for h in hotels.values():
        assert (h.book >= 0).all()


@pytest.fixture(scope="module")
def results(hotels):
    return run(hotels)


def test_saved_report_matches_the_code(results):
    _, summary = results
    saved = json.loads((REPORTS / "results.json").read_text())
    fresh = summary.round(2).to_dict(orient="records")
    assert len(saved) == len(fresh)
    for s, f in zip(saved, fresh):
        for key in ("weekday_average", "pace", "gradient_boosting"):
            assert s[key] == pytest.approx(f[key], abs=0.05), (s["hotel"], s["horizon"], key)


def test_booking_pace_beats_the_weekday_average_at_every_horizon(results):
    _, summary = results
    assert (summary.pace < summary.weekday_average).all()


def test_test_rows_use_only_test_days(hotels):
    rows = build_rows(hotels, *TEST, horizons=[7])
    assert rows.day.min() >= TEST[0] and rows.day.max() <= TEST[1]
    assert set(rows.horizon) == {7}
