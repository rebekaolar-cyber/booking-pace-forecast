"""Interactive demo: how many guests will arrive, and how much does the reservation book help?

Run locally:  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pace import ml  # noqa: E402
from pace.backtest import TEST, TRAIN, build_rows, summarise  # noqa: E402
from pace.data import load_hotels  # noqa: E402
from pace.models import reference_days  # noqa: E402

REPO = "https://github.com/rebekaolar-cyber/booking-pace-forecast"
COLORS = {"Actual guests": "#0b0b0b", "Booking pace (rule)": "#2a78d6",
          "Gradient boosting (ML)": "#eb6834", "Weekday average": "#1baf7a"}
COLUMNS = {"actual": "Actual guests", "pace": "Booking pace (rule)",
           "gradient_boosting": "Gradient boosting (ML)", "weekday_average": "Weekday average"}

st.set_page_config(page_title="Booking-pace forecast", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Rebuilding the reservation book from 119,390 bookings…")
def hotels():
    return load_hotels()


@st.cache_data(show_spinner="Replaying six months of forecasts…")
def backtest():
    h = hotels()
    model = ml.train(build_rows(h, *TRAIN, horizons=range(1, 31)))
    test = build_rows(h, *TEST, horizons=range(1, 31))
    test["gradient_boosting"] = ml.predict(model, test)
    return test


test = backtest()

st.title("How many guests are coming?")
st.markdown(
    "A demand forecast that **reads the reservation book**: start from the usual number for that "
    "weekday, then correct it by how far bookings are running ahead of or behind normal. "
    "Tested on **119,390 real hotel bookings** (two hotels in Portugal, 2015-2017), scored on "
    "six months the models never saw."
)

c1, c2, c3 = st.columns([1.2, 1, 1])
hotel = c1.radio("Hotel", ["City Hotel", "Resort Hotel"], horizontal=True)
horizon = c2.slider("Forecast made … days ahead", 1, 30, 7)
months = c3.select_slider("Months shown", options=pd.date_range(*TEST, freq="MS").strftime("%b %Y").tolist(),
                          value=("May 2017", "Jun 2017"))

g = test[(test.hotel == hotel) & (test.horizon == horizon)].copy()
s = summarise(g).iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average guests arriving per day", f"{s.average_guests:.0f}")
k2.metric("Error: weekday average", f"{s.weekday_average:.1f} guests")
k3.metric("Error: booking pace", f"{s.pace:.1f} guests",
          delta=f"{-s.pace_vs_weekday_pct:.0f}% vs weekday average", delta_color="inverse")
k4.metric("Error: gradient boosting", f"{s.gradient_boosting:.1f} guests",
          delta=f"{-s.ml_vs_weekday_pct:.0f}% vs weekday average", delta_color="inverse")
st.caption("Error = mean absolute error over every test day, 1 Mar - 31 Aug 2017. Lower is better.")

start = pd.Timestamp(pd.to_datetime(months[0], format="%b %Y"))
end = pd.Timestamp(pd.to_datetime(months[1], format="%b %Y")) + pd.offsets.MonthEnd(0)
window = g[g.day.between(start, end)]
long = window.melt(id_vars="day", value_vars=list(COLUMNS), var_name="series", value_name="guests")
long["series"] = long.series.map(COLUMNS)
long["guests"] = long.guests.round(0)
chart = (
    alt.Chart(long)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("guests:Q", title="Guests arriving"),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum.series == "Weekday average", alt.value([4, 2]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("day:T", format="%a %d %b %Y"), "series:N", "guests:Q"],
    )
    .properties(height=340)
    .interactive(bind_y=False)
)
st.subheader("Forecast against what really happened")
st.altair_chart(chart, width="stretch")

st.subheader("Look inside one day")
g["miss"] = (g.weekday_average - g.actual).abs() - (g.pace - g.actual).abs()
default_day = g.sort_values("miss").day.iloc[-1].date()
day = pd.Timestamp(st.date_input("Day", value=default_day, min_value=TEST[0].date(), max_value=TEST[1].date()))
row = g[g.day == day].iloc[0]
h = hotels()[hotel]
refs = reference_days(h, day, horizon)
ks = np.arange(60, -1, -1)
curve = pd.DataFrame({
    "days_before": -ks,
    f"{day:%a %d %b}": [h.booked(day, k) for k in ks],
    f"A normal {day:%A}": [np.mean([h.booked(r, k) for r in refs]) for k in ks],
}).melt(id_vars="days_before", var_name="series", value_name="guests in the book")
left, right = st.columns([2, 1])
line = alt.Chart(curve).mark_line(strokeWidth=2).encode(
    x=alt.X("days_before:Q", title="Days before arrival"),
    y=alt.Y("guests in the book:Q", title="Guests in the reservation book"),
    color=alt.Color("series:N", scale=alt.Scale(domain=[f"{day:%a %d %b}", f"A normal {day:%A}"],
                                                range=["#2a78d6", "#8a8984"]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["series:N", "days_before:Q", alt.Tooltip("guests in the book:Q", format=".0f")],
)
rule = alt.Chart(pd.DataFrame({"x": [-(horizon - 1)]})).mark_rule(strokeDash=[2, 2], color="#52514e").encode(x="x:Q")
left.altair_chart((line + rule).properties(height=300), width="stretch")
right.markdown(
    f"""
**Standing {horizon} day{'s' if horizon > 1 else ''} before {day:%A %d %B}:**

1. The last {len(refs)} {day:%A}s averaged **{row.weekday_average:.0f}** guests.
2. The book held **{row.booked_now}** guests; on a normal {day:%A} it held
   **{row.booked_normally:.0f}** at this point.
3. So bookings are running **{row.booked_now - row.booked_normally:+.0f}** against normal,
   and the pace forecast says **{row.pace:.0f}** (the correction is capped at ±50%).

The ML model said **{row.gradient_boosting:.0f}**. **{row.actual}** guests arrived.
"""
)

with st.expander("Results at every horizon"):
    full = summarise(test[test.horizon.isin([1, 3, 7, 14, 30])])
    full = full.rename(columns={"horizon": "days ahead", "average_guests": "avg guests",
                                **{k: v for k, v in COLUMNS.items() if k != "actual"},
                                "same_day_last_year": "Same day last year"})
    st.dataframe(full.drop(columns=["days", "pace_vs_weekday_pct", "ml_vs_weekday_pct"]).round(1),
                 hide_index=True, width="stretch")

with st.expander("How it works, and what it doesn't show"):
    st.markdown(
        f"""
- **No peeking at the future.** Each forecast uses only bookings in the book on the morning it is
  made, and only days that were already over. The ML model trains on Jul 2016 - Feb 2017 and is
  scored on Mar - Aug 2017. Tests in the repo check both.
- **Simple beats clever up to a week out.** The two-line rule beats gradient boosting at every
  horizon from 1 to 7 days; ML pulls ahead a month out, where the book is still thin.
- **Hotels book further ahead than restaurants.** The 1-day gains are larger than a restaurant with
  walk-ins would see; the 7- and 14-day numbers are the fairer comparison.
- **Data:** Antonio, de Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22,
  41-49, CC BY 4.0. Hotel and guest identities were removed by the authors.

Code, tests and charts: [{REPO.split('github.com/')[1]}]({REPO})
"""
    )

st.caption(f"Built by Rebeka Olar · [GitHub]({REPO})")
