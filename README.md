# How many guests are coming? Forecasting demand from the reservation book

[![tests](https://github.com/rebekaolar-cyber/booking-pace-forecast/actions/workflows/tests.yml/badge.svg)](https://github.com/rebekaolar-cyber/booking-pace-forecast/actions/workflows/tests.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)

**On 119,390 real hotel bookings, a forecast that reads the reservation book makes 53-68% smaller
errors a week ahead than the usual weekday average.** It is two lines of logic, it beats a
gradient-boosting model up to 7 days out, and every number below can be reproduced with one command.

**▶ [Try the live demo](#live-demo)** · [Results](#results) · [How it works](#how-it-works) · [Run it yourself](#run-it-yourself)

![Forecast error by days ahead](reports/error_by_horizon.png)

## Why this matters

Restaurants and hotels decide every day how many people to schedule and how much to order. Most
plan from a fixed weekly pattern ("Fridays are busy"). The reservation book already knows when a
day is going to be unusual, often weeks in advance. This project measures how much that is worth.

## Results

Mean absolute error in **guests per day**. Lower is better. Forecasts for 1 Mar - 31 Aug 2017,
made using only what was known on the morning of the forecast. The ML model was trained on
Jul 2016 - Feb 2017 and never saw the test months.

**City hotel** (140 guests arriving per day on average)

| Days ahead | Weekday average | Same day last year | **Booking pace (rule)** | Gradient boosting (ML) |
|---:|---:|---:|---:|---:|
| 1  | 26.5 | 35.8 | **4.8**  | 6.4 |
| 7  | 26.5 | 35.8 | **8.6**  | 8.9 |
| 14 | 27.5 | 35.8 | **10.7** | 11.0 |
| 30 | 32.0 | 35.8 | 18.3 | **16.7** |

**Resort hotel** (75 guests arriving per day on average)

| Days ahead | Weekday average | Same day last year | **Booking pace (rule)** | Gradient boosting (ML) |
|---:|---:|---:|---:|---:|
| 1  | 21.5 | 28.9 | **6.6** | 6.7 |
| 7  | 21.5 | 28.9 | **10.2** | 10.6 |
| 14 | 22.2 | 28.9 | 13.1 | **12.3** |
| 30 | 21.6 | 28.9 | 16.4 | **14.3** |

All horizons: [`reports/results.md`](reports/results.md).

**What I took from it**

1. **The book is the signal.** Correcting for booking pace cuts the error by 69-82% one day out and
   still by 24-43% a month out.
2. **Simple beats clever up to a week.** The two-line rule beats gradient boosting at every
   horizon from 1 to 7 days, at both hotels. A manager can check it by hand, which matters when it decides who gets a shift.
3. **ML earns its place further out.** A month ahead the book is still thin, and the model's use
   of seasonality and last year pays off: 9-13% better than the rule at 30 days.

![Forecast against actual arrivals](reports/forecast_vs_actual.png)

## How it works

For a day `d`, standing `h` days before it:

1. **Weekday average.** Average the guests on the last 6 same weekdays that were already over.
2. **Booking pace.** Count the guests in the book for `d` right now. Compare that with what was in
   the book for those 6 earlier days at the same distance before them.
3. **Correct.** Add the difference to the average, capped at ±50% so a single odd group booking
   can't run away with the forecast.

![How the booking curve gave the answer away](reports/booking_curve.png)

*28 April 2017: a large group was in the book weeks ahead. The weekday average said 162, the
booking pace said 243, and 241 guests came.*

The reservation book is rebuilt from the raw bookings as a `day × days-before` matrix with a
difference array (`src/pace/data.py`), so a booking counts from the day after it was made until it
is cancelled, and the whole 2-year history builds in under a second.

## Proof it's honest

Backtests leak the future easily. These guard against it, and run on every push:

- **The book only shows bookings that existed at forecast time.** A booking made on the forecast
  day itself doesn't change that forecast (`tests/test_book.py`).
- **Cancellations leave the book on their cancellation date. No-shows stay in the book but don't
  count as guests.** Each has a test.
- **Strict time split:** training ends 28 Feb 2017, testing starts 1 Mar 2017.
- **The published numbers are what the code produces.** A test reruns the whole backtest and fails
  if `reports/results.json` differs.

## Live demo

The interactive version lets you pick a hotel, how far ahead to forecast and any day, and shows the
booking curve behind the number.

![Demo](docs/demo.png)
![Looking inside one day](docs/demo-day.png)

Run it locally with `streamlit run streamlit_app.py`.

## Run it yourself

```bash
git clone https://github.com/rebekaolar-cyber/booking-pace-forecast
cd booking-pace-forecast
pip install -r requirements.txt pytest

PYTHONPATH=src python -m pace.backtest   # rebuilds reports/results.md (about 10 seconds)
PYTHONPATH=src python -m pace.charts     # redraws the three charts
python -m pytest                         # 17 tests
streamlit run streamlit_app.py           # the demo
```

```
src/pace/
  data.py       load bookings, rebuild the reservation book per day
  models.py     weekday average, same day last year, booking-pace rule
  ml.py         gradient boosting on the same signals (scikit-learn)
  backtest.py   replay Mar-Aug 2017 at 1-30 days ahead, write reports/
  charts.py     the README charts
tests/          book rebuilding, no-leakage checks, reproducible results
streamlit_app.py
```

## Limits

- **These are hotels, not restaurants.** Hotel guests book further ahead and rarely walk in, so
  the 1-day gain is bigger than a restaurant with walk-ins would see. The 7- and 14-day numbers are
  the fairer comparison.
- **Two hotels, one country, 2015-2017.** The method transfers; the exact percentages may not.
- **Arrivals, not occupancy.** The target is guests arriving per day, the closest equivalent to
  covers per evening in a restaurant.

## Where this is used

The same rule runs every night in a staffing and scheduling system I built for a 254-seat
restaurant in Enschede. There it turns the forecast into staff per role using the manager's own
rules, with a staff app for schedules, shift swaps and clocking in. That repo is private because it
belongs to the restaurant. Screenshots are available on request.

## Data

Antonio, N., de Almeida, A. & Nunes, L. (2019). *Hotel booking demand datasets.* Data in Brief, 22,
41-49. [doi:10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126). Licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The authors removed all hotel and
customer identifiers. `data/hotels.csv.gz` is the unmodified CSV, gzipped.

Code: MIT licence.

---

Built by **Rebeka Olar**, Business Information Technology student at the University of Twente (NL).
Open to remote roles in data, analytics and automation.
