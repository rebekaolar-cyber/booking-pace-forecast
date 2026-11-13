"""Replay the past: forecast every day of the test period at several horizons, then score.

    python -m pace.backtest            # writes reports/results.json and reports/results.md

Training days:  1 Jul 2016 - 28 Feb 2017 (the model learns here)
Test days:      1 Mar 2017 - 31 Aug 2017 (nothing here is seen before it is scored)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pace import ml
from pace.data import Hotel, load_hotels
from pace.models import pace_forecast, signals

HORIZONS = [1, 3, 7, 14, 30]
TRAIN = (pd.Timestamp("2016-07-01"), pd.Timestamp("2017-02-28"))
TEST = (pd.Timestamp("2017-03-01"), pd.Timestamp("2017-08-31"))
METHODS = {
    "weekday_average": "Weekday average",
    "same_day_last_year": "Same day last year",
    "pace": "Booking pace (rule)",
    "gradient_boosting": "Gradient boosting (ML)",
}
REPORTS = Path(__file__).resolve().parents[2] / "reports"


def build_rows(hotels: dict[str, Hotel], start: pd.Timestamp, end: pd.Timestamp,
               horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for name, hotel in hotels.items():
        for day in pd.date_range(start, end):
            for h in horizons:
                s = signals(hotel, day, h)
                if s is None:
                    continue
                rows.append({
                    "hotel": name, "day": day, "horizon": h,
                    "actual": int(hotel.actual[day]),
                    "weekday_average": s.weekday_average,
                    "same_day_last_year": s.same_day_last_year,
                    "booked_now": s.booked_now,
                    "booked_normally": s.booked_normally,
                    "pace": pace_forecast(s),
                })
    return pd.DataFrame(rows)


def run(hotels: dict[str, Hotel] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (every forecast with its actual, error summary per hotel and horizon)."""
    hotels = hotels or load_hotels()
    train_rows = build_rows(hotels, *TRAIN, horizons=range(1, 31))
    test = build_rows(hotels, *TEST)
    model = ml.train(train_rows)
    test["gradient_boosting"] = ml.predict(model, test)
    return test, summarise(test)


def summarise(test: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (hotel, h), g in test.groupby(["hotel", "horizon"]):
        row = {"hotel": hotel, "horizon": h, "days": len(g), "average_guests": g.actual.mean()}
        for m in METHODS:
            ok = g[m].notna()
            row[m] = float(np.mean(np.abs(g[m][ok] - g.actual[ok])))
        out.append(row)
    s = pd.DataFrame(out)
    s["pace_vs_weekday_pct"] = 100 * (1 - s.pace / s.weekday_average)
    s["ml_vs_weekday_pct"] = 100 * (1 - s.gradient_boosting / s.weekday_average)
    return s


def write_reports(summary: pd.DataFrame, path: Path = REPORTS) -> None:
    path.mkdir(exist_ok=True)
    (path / "results.json").write_text(json.dumps(
        summary.round(2).to_dict(orient="records"), indent=2) + "\n")
    lines = ["# Backtest results", "",
             "Mean absolute error in guests per day (lower is better). "
             f"Test days {TEST[0]:%d %b %Y} - {TEST[1]:%d %b %Y}; "
             "the ML model was trained only on earlier days.", ""]
    for hotel, g in summary.groupby("hotel"):
        lines += [f"## {hotel} (average {g.average_guests.iloc[0]:.0f} guests arriving per day)", "",
                  "| Days ahead | " + " | ".join(METHODS.values()) + " | Pace rule: error cut vs weekday average |",
                  "|---" * (len(METHODS) + 2) + "|"]
        for _, r in g.iterrows():
            cells = " | ".join(f"**{r[m]:.1f}**" if r[m] == min(r[k] for k in METHODS) else f"{r[m]:.1f}"
                               for m in METHODS)
            lines.append(f"| {r.horizon} | {cells} | {r.pace_vs_weekday_pct:.0f}% |")
        lines.append("")
    (path / "results.md").write_text("\n".join(lines))


def main() -> None:
    test, summary = run()
    write_reports(summary)
    test.to_csv(REPORTS / "forecasts.csv.gz", index=False)
    print((REPORTS / "results.md").read_text())


if __name__ == "__main__":
    main()
