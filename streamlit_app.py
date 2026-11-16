"""Plan the next two weeks from your reservation book, and see how accurate that is.

Run locally:  streamlit run streamlit_app.py
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # agent/ and app_pages/ for the assistant tab

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app_pages import ui  # noqa: E402
from pace import ml  # noqa: E402
from pace import plan as P  # noqa: E402
from pace.backtest import TEST, TRAIN, build_rows, summarise  # noqa: E402
from pace.data import load_hotels  # noqa: E402
from pace.models import reference_days  # noqa: E402

REPO = "https://github.com/rebekaolar-cyber/booking-pace-forecast"
UNUSUAL = 15  # % away from the usual weekday that counts as worth a look

st.set_page_config(page_title="Guest forecast", layout="wide",
                   page_icon=str(Path(__file__).parent / "static" / "favicon.png"))
ui.inject_css()


# ---------------------------------------------------------------- loading -------------------

@st.cache_data(show_spinner=False)
def sample_raw():
    return P.sample_export()


@st.cache_data(show_spinner=False)
def sample_actual():
    return P.sample_outcome()


@st.cache_data(show_spinner=False)
def read_table(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Couldn't read this file. Save it as CSV (UTF-8) or Excel and try again.")


@st.cache_data(show_spinner="Checking the forecast on your own last 8 weeks…")
def run_plan(prep_df: pd.DataFrame, until: pd.Timestamp, actual, days: int, skip_closed: bool):
    prep = P.Prepared(prep_df, dropped=0)
    check = P.self_check(prep, until, actual, skip_closed=skip_closed)
    book = P.build_book(prep, until, actual)
    typical = dict(zip(check.days_ahead, check.pace_p80)) if len(check) else None
    return P.plan(book, until, days=days, skip_closed=skip_closed, typical_error=typical), check


def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Plan")
    return buf.getvalue()


# ---------------------------------------------------------------- page ----------------------

st.html(ui.hero(REPO))

plan_tab, proof_tab, ask_tab = st.tabs(["Plan the next two weeks", "How accurate is it?",
                                        "Ask the assistant"])


def day_label(d: pd.Timestamp) -> str:
    return f"{d:%a} {d.day} {d:%b}"


def plan_page():
    source = st.segmented_control(
        "Reservations",
        ["Try it with sample data", "Use my own reservations"],
        default="Try it with sample data", required=True,
        help="The sample is a real city hotel's bookings, exported as they stood on 31 May 2017.",
    )

    actual = None
    if source == "Try it with sample data":
        raw = sample_raw()
        mapping = P.guess_columns(raw.columns)
        dayfirst = True
        st.caption(":material/history: **Sample:** a real 140-guests-a-day city hotel, with its bookings "
                   "as they stood on **31 May 2017**. Because this is history, the plan also shows what "
                   "really happened.")
    else:
        with st.expander("What file do I need?", expanded=False):
            st.markdown(
                "Export your reservations from your booking system (TheFork, Zenchef, Resengo, "
                "OpenTable, your PMS…) as **CSV or Excel**, one row per reservation, covering at "
                "least the last **3 months** and everything booked ahead.\n\n"
                "| Column | Needed | Example |\n|---|---|---|\n"
                "| Day of the visit | yes | 14-11-2026 |\n"
                "| Number of guests | yes | 4 |\n"
                "| Day the reservation was made | yes | 01-11-2026 |\n"
                "| Day it was cancelled | better | 12-11-2026 |\n"
                "| Status (booked, cancelled, no-show, walk-in) | better | Cancelled |\n\n"
                "Walk-ins: add them as reservations made on the day itself, or upload your daily "
                "guest totals from the till below. **Leave out names, phone numbers and e-mails**: "
                "they aren't needed, and this app doesn't save your file anywhere."
            )
            st.download_button("Download a template", P.TEMPLATE.to_csv(index=False).encode(),
                               "reservations_template.csv", "text/csv")
        up = st.file_uploader("Your reservations (CSV or Excel)", type=["csv", "txt", "xlsx", "xls"])
        if up is None:
            return
        try:
            raw = read_table(up.getvalue(), up.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read this file: {e}")
            return
        guess = P.guess_columns(raw.columns)
        need = any(guess[f] is None for f in P.REQUIRED)
        with st.expander("Columns", expanded=need):
            if need:
                st.warning("I couldn't find every column I need. Pick them here.")
            cols = st.columns(5)
            options = [None] + list(raw.columns)
            mapping = {}
            for c, (f, label) in zip(cols, P.FIELDS.items()):
                mapping[f] = c.selectbox(label, options, index=options.index(guess[f]),
                                         format_func=lambda x: "—" if x is None else str(x), key=f"map_{f}")
            dayfirst = st.toggle("Dates are written day-month-year (31-12-2026)", value=True)
            st.dataframe(raw.head(5), hide_index=True)
        with st.expander("Optional: real guests per day from your till (includes walk-ins)"):
            till = st.file_uploader("Daily totals (one row per day: date, guests)", type=["csv", "txt", "xlsx", "xls"],
                                    key="till")
            if till is not None:
                t = read_table(till.getvalue(), till.name)
                c1, c2 = st.columns(2)
                dcol = c1.selectbox("Date column", list(t.columns), key="till_d")
                gcol = c2.selectbox("Guests column", list(t.columns), index=min(1, len(t.columns) - 1), key="till_g")
                actual = P.daily_totals(t, dcol, gcol, dayfirst)

    try:
        prep = P.prepare(raw, mapping, dayfirst)
    except ValueError as e:
        st.error(str(e))
        return
    res = prep.reservations

    last_made = res.made_on.max()
    default_until = P.SAMPLE_EXPORT_DAY if source.startswith("Try") else last_made
    c1, c2, c3 = st.columns([1, 1, 1.4])
    until = pd.Timestamp(c1.date_input(
        "Bookings exported on", value=default_until.date(),
        min_value=res.made_on.min().date(), max_value=last_made.date(),
        help="The forecast starts the day after. Bookings made after this day are ignored."))
    days = c2.selectbox("Days to plan", [7, 14, 21, 28], index=1)
    skip_closed = c3.toggle("Skip days with no guests (closed days)", value=True,
                            help="A closed day (holiday, renovation) won't drag the weekday average down.")

    history_days = (until - res.visit_date.min()).days
    if history_days < 21:
        st.error(f"Your file only goes back {history_days} days before {until:%d %b %Y}. "
                 "The forecast needs at least 3 weeks of past visits, ideally 3 months.")
        return

    p, check = run_plan(res, until, None if actual is None else actual, days, skip_closed)
    if source.startswith("Try"):
        p["actual"] = p.date.map(sample_actual())

    notes = [f"{len(res):,} reservations read"]
    if prep.dropped:
        notes.append(f"{prep.dropped:,} rows skipped (no readable date or no guests)")
    st.caption(" · ".join(notes + prep.notes))

    open_days = p[~p.closed]
    if open_days.empty:
        st.warning("No open days found to forecast. Try switching off 'Skip days with no guests'.")
        return

    # ---- headline figures
    has_actual = "actual" in p
    busiest = open_days.loc[open_days.forecast.idxmax()]
    status = p.vs_usual_pct.map(lambda v: ui.day_status(v, UNUSUAL))
    unusual = open_days[status[open_days.index] != "usual"]
    total, usual_total = open_days.forecast.sum(), open_days.usual.sum()
    change = total / usual_total - 1 if usual_total else 0.0
    if abs(change) < 0.005:
        against_usual = f"about the same as a usual {days} days"
    else:
        against_usual = f"{abs(change):.0%} {'more' if change > 0 else 'fewer'} than a usual {days} days"
    figures = [
        (f"{total:,.0f}", f"guests expected in the next {days} days", against_usual),
        (day_label(busiest.date), "busiest day", f"{busiest.forecast:.0f} guests"),
        (f"{len(unusual)} day{'' if len(unusual) == 1 else 's'}",
         "needs a second look" if len(unusual) == 1 else "need a second look",
         f"{UNUSUAL}% or more away from the usual weekday"),
    ]
    if len(check):
        row7 = check[check.days_ahead == min(7, check.days_ahead.max())].iloc[0]
        figures.append((f"± {row7.pace_error:.0f} guests",
                        f"average miss on your last 8 weeks, {int(row7.days_ahead)} days ahead",
                        f"the weekday average alone: ± {row7.usual_error:.0f}"))
    st.html(ui.stats(figures))

    # ---- the days ahead as a calendar
    tiles = []
    for i, r in p.iterrows():
        tile = {"date": r.date, "closed": bool(r.closed)}
        if not r.closed:
            likely = "" if np.isnan(r.low) else f"likely {r.low:.0f} to {r.high:.0f}"
            came = f"{r.actual:.0f} came" if has_actual and not np.isnan(r.actual) else ""
            title = [f"Forecast: {r.forecast:.0f} guests", likely.capitalize(),
                     f"Usual for a {r.date:%A}: {r.usual:.0f}", f"Booked so far: {r.booked:.0f}",
                     came.replace(" came", " actually came")]
            tile.update(forecast=r.forecast, usual=r.usual, status=status[i],
                        vs_usual_pct=r.vs_usual_pct, sub=came or likely,
                        title=". ".join(t for t in title if t) + ".")
        tiles.append(tile)
    st.html(ui.calendar(tiles))
    st.html(ui.key([
        ("bar", ui.CALM, "about usual"),
        ("bar", ui.HOT, f"busier than usual by {UNUSUAL}% or more"),
        ("bar", ui.COOL, f"quieter by {UNUSUAL}% or more"),
        ("tick", ui.INK, "line across each bar: the usual for that weekday"),
    ]))

    # ---- chart: how sure each day is
    st.subheader("Forecast and likely range")
    c = open_days.copy()
    c["day"] = c.date.map(day_label)
    c["label"] = c.date.map(lambda d: f"{d:%a}|{d.day} {d:%b}")  # two lines on the axis
    c["status"] = status[c.index]
    c["flag"] = [f"{'↑' if s == 'busier' else '↓'} {abs(v):.0f}%" if s != "usual" else ""
                 for s, v in zip(c.status, c.vs_usual_pct)]
    c["label_y"] = c[["forecast", "high", "usual"] + (["actual"] if has_actual else [])].max(axis=1)
    order = list(c.label)
    # Sizes follow the chart's width, so 7 days on a wide screen and 28 on a phone both fit.
    band = f"(width / {len(c)})"
    bar = f"clamp({band} * 0.36, 3, 22)"
    beside = f"clamp({band} * 0.33, 6, {bar} / 2 + 10)"
    dot = f"clamp({band} * 1.4, 24, 64)"
    base = alt.Chart(c).encode(x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(
        labelAngle=0, labelExpr="split(datum.label, '|')", labelLineHeight=15, ticks=False, labelPadding=8)))
    tooltip = [alt.Tooltip("day:N", title="Day"),
               alt.Tooltip("forecast:Q", title="Forecast", format=".0f"),
               alt.Tooltip("low:Q", title="Likely from", format=".0f"),
               alt.Tooltip("high:Q", title="Likely to", format=".0f"),
               alt.Tooltip("usual:Q", title="Usual for this weekday", format=".0f"),
               alt.Tooltip("booked:Q", title="Booked so far")]
    if has_actual:
        tooltip.append(alt.Tooltip("actual:Q", title="Actually came"))
    layers = [
        base.mark_bar(size=alt.expr(bar), cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            y=alt.Y("forecast:Q", title=None),
            color=alt.Color("status:N", legend=None,
                            scale=alt.Scale(domain=list(ui.STATUS), range=list(ui.STATUS.values())))),
        # Each dark mark sits on a white one a little larger, so it stays visible over a bar.
        base.mark_tick(color="white", thickness=5, size=alt.expr(f"{bar} + 12")).encode(y="usual:Q"),
        base.mark_tick(color=ui.INK, thickness=2, size=alt.expr(f"{bar} + 10")).encode(y="usual:Q"),
        base.mark_rule(color="white", strokeWidth=4.5).encode(y="low:Q", y2="high:Q"),
        base.mark_rule(color=ui.INK, strokeWidth=1.5).encode(y="low:Q", y2="high:Q"),
        base.mark_point(shape="circle", filled=True, fill="white", stroke=ui.INK, strokeWidth=1.75,
                        opacity=1, size=alt.expr(dot), xOffset=alt.expr(f"-{beside}")).encode(y="booked:Q"),
    ]
    if has_actual:
        layers.append(base.mark_point(shape="diamond", filled=True, fill=ui.INK, opacity=1,
                                      size=alt.expr(f"{dot} * 1.25"), xOffset=alt.expr(beside)).encode(y="actual:Q"))
    layers.append(base.mark_text(dy=-10, fontSize=12, fontWeight=600, color=ui.INK2).encode(
        y="label_y:Q", text="flag:N"))
    # One wide invisible bar per day on top, so the tooltip opens anywhere over that day.
    layers.append(base.mark_bar(opacity=0, size=alt.expr(f"{band} * 0.96")).encode(
        y=alt.Y("label_y:Q"), tooltip=tooltip))
    marks = [("bar", ui.CALM, "forecast"), ("whisker", ui.INK, "likely range (8 in 10 days)"),
             ("tick", ui.INK, "usual for that weekday"), ("ring", ui.INK, "booked so far")]
    if has_actual:
        marks.append(("diamond", ui.INK, "what really happened"))
    st.html(ui.key(marks))
    st.altair_chart(ui.polish(alt.layer(*layers), height=330), width="stretch")

    # ---- staffing + table
    st.subheader("Staffing plan")
    with st.expander("Staffing rules: change these to your own", expanded=False):
        r1, r2 = st.columns([2, 1])
        rules = r1.data_editor(
            P.DEFAULT_RULES, num_rows="dynamic", hide_index=True, key="rules", width="stretch",
            column_config={
                "role": st.column_config.TextColumn("Role"),
                "guests_per_person": st.column_config.NumberColumn("Guests per person", min_value=1, step=1),
                "minimum": st.column_config.NumberColumn("Minimum on an open day", min_value=0, step=1),
            })
        r2.caption("People = guests ÷ guests per person, rounded up, never below the minimum. "
                   "Add a row for every role you schedule.")
        staff_on = r2.radio("Staff for", ["The forecast", "The top of the likely range"],
                            help="Plan for the top of the range when running short costs more than one extra person.")
    staff = P.staffing(p.assign(high=p.high.fillna(p.forecast)), rules,
                       use="forecast" if staff_on == "The forecast" else "high")

    def flag(r):
        if r.closed:
            return "Closed"
        s = ui.day_status(r.vs_usual_pct, UNUSUAL)
        if s != "usual":
            return f"{'↑' if s == 'busier' else '↓'} {abs(r.vs_usual_pct):.0f}% {s}"
        return ""

    table = pd.DataFrame({
        "Date": p.date.map(day_label),
        "Booked so far": p.booked,
        "Usual": p.usual.round(),
        "Forecast": p.forecast.round(),
        "Likely range": [("" if np.isnan(lo) else f"{lo:.0f}-{hi:.0f}") for lo, hi in zip(p.low, p.high)],
        "vs usual": p.apply(flag, axis=1),
    })
    if has_actual:
        table["Actually came"] = p.actual
    for col in staff.columns[1:]:
        table[col] = staff[col]

    def flag_colour(v: str) -> str:
        return ("background-color: #FCE4DA" if v.startswith("↑")
                else "background-color: #DEEAF7" if v.startswith("↓") else "")

    numbers = [col for col in table.columns if table[col].dtype.kind in "if"]
    shown = table.style.map(flag_colour, subset=["vs usual"]).format("{:.0f}", subset=numbers, na_rep="")
    st.dataframe(shown, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
                 column_config={
                     "Forecast": st.column_config.ProgressColumn(
                         format="%d", min_value=0, max_value=float(np.nanmax([p.forecast.max(), 1])),
                         color=ui.CALM, width="medium"),
                     "vs usual": st.column_config.TextColumn(width="medium"),
                 })
    with st.container(horizontal=True):
        st.download_button("Download Excel", to_excel(table), f"guest_plan_{until:%Y%m%d}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/download:")
        st.download_button("Download CSV", table.to_csv(index=False).encode(),
                           f"guest_plan_{until:%Y%m%d}.csv", "text/csv", icon=":material/download:")

    if has_actual:
        hit = open_days.assign(actual=p.actual).dropna(subset=["actual"])
        mae_p = (hit.forecast - hit.actual).abs().mean()
        mae_u = (hit.usual - hit.actual).abs().mean()
        inside = ((hit.actual >= hit.low) & (hit.actual <= hit.high)).mean()
        st.success(f"**How this sample turned out:** the forecast was off by **{mae_p:.0f}** guests a day "
                   f"on average (the usual weekday number: {mae_u:.0f}), and **{inside:.0%}** of these days "
                   "landed inside the likely range. Over 50 different two-week plans on this data "
                   "(every other week, Sep 2016 - Aug 2017, both hotels), 79-80% of days landed inside it, "
                   "so a single fortnight can come out better or worse.")

    with st.expander("How sure is it, on your own data?"):
        if len(check):
            ch = check.rename(columns={"days_ahead": "Days ahead", "days_scored": "Days checked",
                                       "pace_error": "Forecast off by (avg)",
                                       "usual_error": "Usual weekday off by (avg)",
                                       "pace_p80": "8 in 10 days within"}).round(1)
            st.dataframe(ch, hide_index=True)
            st.caption("Replayed on your last 8 weeks: for each past day the reservation book is rebuilt "
                       "as it stood back then, so the check never uses information it couldn't have had.")
        else:
            st.write("Not enough history yet to check (needs about 10 weeks).")


with plan_tab:
    plan_page()


# ---------------------------------------------------------------- proof tab -----------------

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


SERIES = {  # column -> (name, colour, line width, dash, glyph in the key)
    "weekday_average": ("Weekday average", ui.BASELINE, 1.75, [5, 3], "dash"),
    "gradient_boosting": ("Gradient boosting (ML)", ui.ML, 1.75, [1, 0], "thin"),
    "pace": ("Booking pace (rule)", ui.PACE, 2.75, [1, 0], "line"),
    "actual": ("Actual guests", ui.INK, 2.75, [1, 0], "line"),
}
COLUMNS = {column: name for column, (name, *_) in SERIES.items()}


def against_baseline(pct: float) -> str:
    return f"{abs(pct):.0f}% {'lower' if pct >= 0 else 'higher'} than the weekday average"


with proof_tab:
    st.markdown(
        "Tested on **119,390 real hotel bookings** (two hotels in Portugal, 2015-2017), scored on six "
        "months the models never saw. Every forecast uses only bookings that existed on the morning "
        "it was made."
    )
    test = backtest()
    c1, c2, c3 = st.columns([1.2, 1, 1])
    hotel = c1.segmented_control("Hotel", ["City Hotel", "Resort Hotel"], default="City Hotel", required=True)
    horizon = c2.slider("Forecast made … days ahead", 1, 30, 7)
    months = c3.select_slider("Months shown",
                              options=pd.date_range(*TEST, freq="MS").strftime("%b %Y").tolist(),
                              value=("May 2017", "Jun 2017"))
    g = test[(test.hotel == hotel) & (test.horizon == horizon)].copy()
    s = summarise(g).iloc[0]

    st.subheader("How far off, in guests per day")
    st.html(ui.error_bars([
        ("Weekday average", s.weekday_average, ui.BASELINE, "what most managers do by feel"),
        ("Booking pace (rule)", s.pace, ui.PACE, against_baseline(s.pace_vs_weekday_pct)),
        ("Gradient boosting (ML)", s.gradient_boosting, ui.ML, against_baseline(s.ml_vs_weekday_pct)),
    ]))
    st.caption(f"Mean absolute error over every test day, 1 Mar - 31 Aug 2017. Lower is better. "
               f"This hotel averages {s.average_guests:.0f} guests arriving per day.")

    start = pd.Timestamp(pd.to_datetime(months[0], format="%b %Y"))
    end = pd.Timestamp(pd.to_datetime(months[1], format="%b %Y")) + pd.offsets.MonthEnd(0)
    window = g[g.day.between(start, end)]
    long = window.melt(id_vars="day", value_vars=list(COLUMNS), var_name="series", value_name="guests")
    long["series"] = long.series.map(COLUMNS)
    long["guests"] = long.guests.round(0)
    lines = [  # drawn in this order, so the two that matter most end up on top
        alt.Chart(long[long.series == name]).mark_line(
            color=colour, strokeWidth=width, strokeDash=dash, strokeJoin="round").encode(
            x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=10)),
            y=alt.Y("guests:Q", title=None),
            tooltip=[alt.Tooltip("day:T", format="%a %d %b %Y"), "series:N", "guests:Q"])
        for name, colour, width, dash, _ in SERIES.values()
    ]
    st.subheader("Forecast against what really happened")
    st.html(ui.key([(glyph, colour, name) for name, colour, _, _, glyph in reversed(SERIES.values())]))
    st.altair_chart(ui.polish(alt.layer(*lines).interactive(bind_y=False), height=340), width="stretch")

    st.subheader("Look inside one day")
    g["miss"] = (g.weekday_average - g.actual).abs() - (g.pace - g.actual).abs()
    default_day = g.sort_values("miss").day.iloc[-1].date()
    day = pd.Timestamp(st.date_input("Day", value=default_day, min_value=TEST[0].date(),
                                     max_value=TEST[1].date(), width=260))
    row = g[g.day == day].iloc[0]
    h = hotels()[hotel]
    refs = reference_days(h, day, horizon)
    ks = np.arange(60, -1, -1)
    this_day, normal_day = f"{day:%a} {day.day} {day:%b}", f"A normal {day:%A}"
    curve = pd.DataFrame({
        "days_before": -ks,
        this_day: [h.booked(day, k) for k in ks],
        normal_day: [np.mean([h.booked(r, k) for r in refs]) for k in ks],
    }).melt(id_vars="days_before", var_name="series", value_name="guests in the book")
    # Same colours as the plan: orange-red when the book runs ahead of normal, blue when behind.
    ahead = row.booked_now - row.booked_normally
    this_colour = ui.HOT if ahead > 0 else ui.COOL if ahead < 0 else ui.CALM
    left, right = st.columns([2, 1])
    left.html(ui.key([("line", this_colour, this_day), ("thin", ui.BASELINE, normal_day),
                      ("dash", ui.INK2, "the day the forecast was made")]))
    tip = ["series:N", alt.Tooltip("days_before:Q", title="days before arrival"),
           alt.Tooltip("guests in the book:Q", format=".0f")]
    axes = dict(x=alt.X("days_before:Q", title="Days before arrival", axis=alt.Axis(tickCount=12)),
                y=alt.Y("guests in the book:Q", title=None))
    one_day = alt.layer(
        alt.Chart(curve[curve.series == normal_day]).mark_line(color=ui.BASELINE, strokeWidth=2).encode(tooltip=tip, **axes),
        alt.Chart(curve[curve.series == this_day]).mark_line(color=this_colour, strokeWidth=3,
                                                              strokeJoin="round").encode(tooltip=tip, **axes),
        alt.Chart(pd.DataFrame({"x": [-(horizon - 1)]})).mark_rule(strokeDash=[4, 3], color=ui.INK2).encode(x="x:Q"),
    )
    left.altair_chart(ui.polish(one_day, height=300), width="stretch")
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

@st.cache_resource(show_spinner=False)
def sample_backend():
    """The assistant's tools on the sample export: the same numbers as the Plan tab's sample."""
    from agent.real_backend import RealBackend
    return RealBackend()


with ask_tab:
    from app_pages.assistant import render as render_assistant

    st.info("The assistant answers from the **sample hotel's** forecast (export of 31 May 2017, so "
            "**today is Thursday 1 June 2017**) with the default staffing rules. Every number it "
            "gives comes from the same tools as the Plan tab; it never sees your uploaded file.",
            icon=":material/info:")
    render_assistant(sample_backend())

st.html(ui.footer(REPO))
