# Backtest results

Mean absolute error in guests per day (lower is better). Test days 01 Mar 2017 - 31 Aug 2017; the ML model was trained only on earlier days.

## City Hotel (average 140 guests arriving per day)

| Days ahead | Weekday average | Same day last year | Booking pace (rule) | Gradient boosting (ML) | Pace rule: error cut vs weekday average |
|---|---|---|---|---|---|
| 1 | 26.5 | 35.8 | **4.8** | 6.4 | 82% |
| 3 | 26.5 | 35.8 | **7.2** | 8.0 | 73% |
| 7 | 26.5 | 35.8 | **8.6** | 8.9 | 68% |
| 14 | 27.5 | 35.8 | **10.7** | 11.0 | 61% |
| 30 | 32.0 | 35.8 | 18.3 | **16.7** | 43% |

## Resort Hotel (average 75 guests arriving per day)

| Days ahead | Weekday average | Same day last year | Booking pace (rule) | Gradient boosting (ML) | Pace rule: error cut vs weekday average |
|---|---|---|---|---|---|
| 1 | 21.5 | 28.9 | **6.6** | 6.7 | 69% |
| 3 | 21.5 | 28.9 | **8.5** | 9.1 | 61% |
| 7 | 21.5 | 28.9 | **10.2** | 10.6 | 53% |
| 14 | 22.2 | 28.9 | 13.1 | **12.3** | 41% |
| 30 | 21.6 | 28.9 | 16.4 | **14.3** | 24% |
