# Repeated live eval, 19 Nov 2026

**Question:** if the same questions are asked several times, how much does the score move?
One run can't tell a real regression from a model having an off day.

**Setup:** `claude-haiku-5-5`, prompt v2, real forecast, a fixed sample of 20 of the 50 questions
(spread over the categories), each asked 3 times. GitHub Actions `eval` workflow, run by hand
with `sample=20, repeat=3`.

- Run [44467599747](https://github.com/rebekaolar-cyber/booking-pace-forecast/actions/runs/44467599747),
  scorer at commit 1862d5f.
- Run [44467345406](https://github.com/rebekaolar-cyber/booking-pace-forecast/actions/runs/44467345406),
  the same questions, started two minutes earlier, scorer one commit older (before the "Fri 23"
  date fix).

## Results

| | Run 44467599747 | Run 44467345406 |
|---|---|---|
| Answers (20 questions x 3) | 60 | 60 |
| Pass rate per repeat | 85%, 75%, 85% | 85%, 80%, 100% |
| Mean pass rate | 82% | 88% |
| Questions that always passed | 13 of 20 | 14 of 20 |
| Questions that never passed | oos07 | none |
| Questions that passed only sometimes | dr01, dr03, dr05, nl03, nl05, ta02 | nl03, nl05, nl08, oos05, st08, ta02 |
| Every number traced | 53/60 | 54/60 |
| Right tools | 57/60 | 59/60 |
| Refused when it should | 9/12 | 10/12 |
| Question's language | 59/60 | 60/60 |
| Errors | 0 | 0 |
| Latency p95 | 5.804 s | 4.928 s |
| Cost per 100 questions | $0.0869 | $0.0835 |

**What this shows:** the same 20 questions scored anywhere from 75% to 100% from one repeat to the
next. Earlier 10-question runs scored 90-100%. A difference of 10 to 15 points between two
single runs is therefore within run-to-run noise, not proof that a change helped or hurt.

## Hand check of the 11 flags in run 44467599747

| Flag | What the answer said | Verdict |
|---|---|---|
| dr05, language | An English question answered in Dutch | **Real mistake** |
| oos07 (3 times), tools and refusal | Declined to book a table, but also looked up and quoted the day's forecast | **Against the spec** (a refusal should not call tools); no wrong numbers |
| ta02, "about 26" | The weekday average's error was 26.5 | Unclear: rounding 26.5 down |
| nl05, "80%" | "lager dan de 80% waar het model op rekent" (the range's target) | Scorer false alarm |
| nl03 (2 times), "15%" | The flag threshold, "binnen 15%" / "meer dan 15%" | Scorer false alarm |
| ta02, "8 in 10" | "In the back-test, 8 in 10 days landed inside it" (the range's definition) | Scorer false alarm |
| dr03, "two" | "The two ranges overlap" | Scorer false alarm |
| dr01, "5" | "Mon 5, Wed 7, Sat 10 and Sun 11 are the flagged days" | Scorer false alarm, **fixed** after this run (a weekday and day number in a list) |

So of the 11 flagged answers: 1 real mistake, 3 against the refusal spec, 1 unclear and 6 scorer
false alarms. None had a wrong or invented number. The 49 answers that passed every check were not
read by hand, so a mistake the scorer can't see (see Known limits in the README) could be among them.

## Scorer changes made on 19 Nov because of these runs

- "Three of the four days are below usual" is a count of days, not a rate (17 Nov dr07 flag).
- "8 in 10 past days landed within ±16 guests" is the range definition (first ta02 flag).
- "Fri 23" with no month is a date when exactly one returned day matches it (dr07, dr01 flags).

Each has tests for the correct answers it now accepts and for the wrong answers it must still
reject (tests/test_scoring.py).
