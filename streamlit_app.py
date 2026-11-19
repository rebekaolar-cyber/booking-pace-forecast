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

from app_pages import i18n as T  # noqa: E402
from app_pages import ui  # noqa: E402
from app_pages.i18n import tr  # noqa: E402
from pace import ml  # noqa: E402
from pace import plan as P  # noqa: E402
from pace.backtest import TEST, TRAIN, build_rows, summarise  # noqa: E402
from pace.data import load_hotels  # noqa: E402
from pace.models import reference_days  # noqa: E402

REPO = "https://github.com/rebekaolar-cyber/booking-pace-forecast"
UNUSUAL = 15  # % away from the usual weekday that counts as worth a look

st.set_page_config(page_title=tr("Guest forecast", "Gastenvoorspelling"), layout="wide",
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
    raise ValueError(tr("Couldn't read this file. Save it as CSV (UTF-8) or Excel and try again.",
                        "Dit bestand kon niet worden gelezen. Sla het op als CSV (UTF-8) of Excel en "
                        "probeer het opnieuw."))


@st.cache_data(show_spinner=False)
def run_plan(prep_df: pd.DataFrame, until: pd.Timestamp, actual, days: int, skip_closed: bool):
    prep = P.Prepared(prep_df, dropped=0)  # the spinner is drawn by the caller, in the visitor's language
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

T.switch()  # English / Nederlands, top right; it only changes this visitor's view, never the link
st.html(ui.hero(REPO))

plan_tab, proof_tab, ask_tab = st.tabs([tr("Plan the next two weeks", "Plan de komende twee weken"),
                                        tr("How accurate is it?", "Hoe nauwkeurig is het?"),
                                        tr("Ask the assistant", "Vraag het de assistent")])


def day_label(d: pd.Timestamp) -> str:
    return T.short_date(d)


def walk_in_settings(raw: pd.DataFrame, mapping: dict, has_till: bool):
    """The restaurant's walk-in estimate: (weekend share, weekday share) of all guests, or None."""
    with st.expander("Walk-ins", expanded=True):
        if has_till:
            st.caption(tr(":material/check: Your till totals already include walk-ins, so none are "
                          "added on top of the forecast.",
                          ":material/check: Je kassatotalen bevatten de walk-ins al, dus er komen er "
                          "geen bovenop de voorspelling."))
            return None
        in_file = P.has_walk_in_rows(raw, mapping)
        add = st.toggle(tr("Add walk-ins on top of the reservations", "Walk-ins bovenop de reserveringen tellen"),
                        value=not in_file, key="walk_in_on",
                        help=tr("Switch off if your file already lists every walk-in.",
                                "Zet dit uit als elke walk-in al in je bestand staat."))
        if in_file:
            st.caption(tr("Your file has rows marked as walk-ins, so they are probably counted already.",
                          "Je bestand heeft rijen gemarkeerd als walk-in, dus die zijn waarschijnlijk al "
                          "meegeteld."))
        if not add:
            return None
        c1, c2 = st.columns(2)
        weekend = c1.number_input(tr("Walk-ins on Fri, Sat and Sun (% of all guests)",
                                     "Walk-ins op vr, za en zo (% van alle gasten)"), 0, 90,
                                  round(100 * P.WALK_IN_SHARE["weekend"]), step=5, key="walk_in_weekend")
        weekday = c2.number_input(tr("Walk-ins Monday to Thursday (% of all guests)",
                                     "Walk-ins maandag t/m donderdag (% van alle gasten)"), 0, 90,
                                  round(100 * P.WALK_IN_SHARE["weekday"]), step=5, key="walk_in_weekday")
        st.caption(tr("40% means 4 in 10 guests walk in without booking: a Saturday with 60 guests "
                      "expected from reservations becomes 100 guests, 40 of them walk-ins. Change the "
                      "numbers to what you see at your place.",
                      "40% betekent dat 4 op de 10 gasten zonder reservering binnenlopen: een zaterdag "
                      "met 60 verwachte gasten uit reserveringen wordt 100 gasten, van wie 40 walk-ins. "
                      "Pas de getallen aan naar wat je in je eigen zaak ziet."))
        return weekend / 100, weekday / 100


def plan_page():
    sources = {"sample": tr("Try it with sample data", "Probeer het met voorbeelddata"),
               "own": tr("Use my own reservations", "Gebruik mijn eigen reserveringen")}
    source = st.segmented_control(
        tr("Reservations", "Reserveringen"), list(sources), format_func=sources.get,
        default="sample", required=True, key="source",
        help=tr("The sample is a real city hotel's bookings, exported as they stood on 31 May 2017.",
                "Het voorbeeld zijn de echte boekingen van een stadshotel, geëxporteerd zoals ze er "
                "op 31 mei 2017 voor stonden."),
    )

    actual = None
    walk_in = None  # (weekend share, weekday share) of all guests, when walk-ins are added on top
    if source == "sample":
        raw = sample_raw()
        mapping = P.guess_columns(raw.columns)
        dayfirst = True
        st.caption(tr(":material/history: **Sample:** a real 140-guests-a-day city hotel, with its "
                      "bookings as they stood on **31 May 2017**. Because this is history, the plan also "
                      "shows what really happened.",
                      ":material/history: **Voorbeeld:** een echt stadshotel met 140 gasten per dag, met "
                      "de boekingen zoals ze er op **31 mei 2017** voor stonden. Omdat dit geschiedenis "
                      "is, laat het plan ook zien wat er echt gebeurde."))
    else:
        kinds = {"Restaurant": "Restaurant", "Hotel": "Hotel"}  # the same word in both languages
        kind = st.segmented_control(
            tr("What kind of place?", "Wat voor zaak?"), list(kinds), format_func=kinds.get,
            default="Restaurant", required=True, key="kind",
            help=tr("Hotel guests almost all book ahead. In a restaurant many guests walk in, so the "
                    "plan adds an estimate for them on top of the reservations.",
                    "Hotelgasten boeken bijna allemaal vooraf. In een restaurant lopen veel gasten "
                    "zomaar binnen, dus het plan telt een schatting voor hen bij de reserveringen op."))
        with st.expander(tr("What file do I need?", "Welk bestand heb ik nodig?"), expanded=False):
            st.markdown(tr(
                "Export your reservations from your booking system (TheFork, Zenchef, Resengo, "
                "OpenTable, your PMS…) as **CSV or Excel**, one row per reservation, covering at "
                "least the last **3 months** and everything booked ahead.\n\n"
                "| Column | Needed | Example |\n|---|---|---|\n"
                "| Day of the visit | yes | 14-11-2026 |\n"
                "| Number of guests | yes | 4 |\n"
                "| Day the reservation was made | yes | 01-11-2026 |\n"
                "| Day it was cancelled | better | 12-11-2026 |\n"
                "| Status (booked, cancelled, no-show, walk-in) | better | Cancelled |\n\n"
                "Walk-ins: add them as reservations made on the day itself, upload your daily "
                "guest totals from the till below, or let the app estimate them (restaurants: "
                "40% of guests on Fri-Sun, 20% Mon-Thu, which you can change). **Leave out names, phone numbers and e-mails**: "
                "they aren't needed, and this app doesn't save your file anywhere.",
                "Exporteer je reserveringen uit je reserveringssysteem (TheFork, Zenchef, Resengo, "
                "OpenTable, je PMS…) als **CSV of Excel**, één rij per reservering, over minstens de "
                "laatste **3 maanden** plus alles wat vooruit geboekt is.\n\n"
                "| Kolom | Nodig | Voorbeeld |\n|---|---|---|\n"
                "| Dag van het bezoek | ja | 14-11-2026 |\n"
                "| Aantal gasten | ja | 4 |\n"
                "| Dag waarop gereserveerd is | ja | 01-11-2026 |\n"
                "| Dag van annulering | beter | 12-11-2026 |\n"
                "| Status (geboekt, geannuleerd, no-show, walk-in) | beter | Geannuleerd |\n\n"
                "Walk-ins: zet ze erin als reserveringen die op de dag zelf gemaakt zijn, upload "
                "hieronder je dagtotalen uit de kassa, of laat de app ze schatten (restaurants: 40% "
                "van de gasten op vr-zo, 20% op ma-do, aan te passen). **Laat namen, telefoonnummers "
                "en e-mailadressen weg**: die zijn niet nodig, en deze app slaat je bestand nergens op."
            ))
            st.download_button(tr("Download a template", "Download een sjabloon"),
                               P.TEMPLATE.to_csv(index=False).encode(),
                               "reservations_template.csv", "text/csv")
        up = st.file_uploader(tr("Your reservations (CSV or Excel)", "Je reserveringen (CSV of Excel)"),
                              type=["csv", "txt", "xlsx", "xls"], key="upload")
        if up is None:
            return
        try:
            raw = read_table(up.getvalue(), up.name)
        except Exception as e:  # noqa: BLE001
            st.error(tr(f"Couldn't read this file: {e}", f"Dit bestand kon niet worden gelezen: {e}"))
            return
        guess = P.guess_columns(raw.columns)
        need = any(guess[f] is None for f in P.REQUIRED)
        with st.expander(tr("Columns", "Kolommen"), expanded=need):
            if need:
                st.warning(tr("I couldn't find every column I need. Pick them here.",
                              "Ik kon niet elke kolom vinden die ik nodig heb. Kies ze hier."))
            cols = st.columns(5)
            options = [None] + list(raw.columns)
            mapping = {}
            for c, (f, label) in zip(cols, P.FIELDS.items()):
                mapping[f] = c.selectbox(T.field_name(label), options, index=options.index(guess[f]),
                                         format_func=lambda x: "—" if x is None else str(x), key=f"map_{f}")
            dayfirst = st.toggle(tr("Dates are written day-month-year (31-12-2026)",
                                    "Datums staan als dag-maand-jaar (31-12-2026)"), value=True, key="dayfirst")
            st.dataframe(raw.head(5), hide_index=True)
        with st.expander(tr("Optional: real guests per day from your till (includes walk-ins)",
                            "Optioneel: echte gasten per dag uit je kassa (inclusief walk-ins)")):
            till = st.file_uploader(tr("Daily totals (one row per day: date, guests)",
                                       "Dagtotalen (één rij per dag: datum, gasten)"),
                                    type=["csv", "txt", "xlsx", "xls"], key="till")
            if till is not None:
                t = read_table(till.getvalue(), till.name)
                c1, c2 = st.columns(2)
                dcol = c1.selectbox(tr("Date column", "Datumkolom"), list(t.columns), key="till_d")
                gcol = c2.selectbox(tr("Guests column", "Gastenkolom"), list(t.columns),
                                    index=min(1, len(t.columns) - 1), key="till_g")
                actual = P.daily_totals(t, dcol, gcol, dayfirst)
        if kind == "Restaurant":
            walk_in = walk_in_settings(raw, mapping, has_till=actual is not None)

    try:
        prep = P.prepare(raw, mapping, dayfirst)
    except ValueError as e:
        st.error(T.message(str(e)))
        return
    res = prep.reservations

    last_made = res.made_on.max()
    default_until = P.SAMPLE_EXPORT_DAY if source == "sample" else last_made
    c1, c2, c3 = st.columns([1, 1, 1.4])
    until = pd.Timestamp(c1.date_input(
        tr("Bookings exported on", "Boekingen geëxporteerd op"), value=default_until.date(),
        min_value=res.made_on.min().date(), max_value=last_made.date(),
        help=tr("The forecast starts the day after. Bookings made after this day are ignored.",
                "De voorspelling begint de dag erna. Boekingen van na deze dag tellen niet mee.")))
    days = c2.selectbox(tr("Days to plan", "Dagen om te plannen"), [7, 14, 21, 28], index=1, key="days")
    skip_closed = c3.toggle(tr("Skip days with no guests (closed days)", "Dagen zonder gasten overslaan (gesloten dagen)"),
                            value=True, key="skip_closed",
                            help=tr("A closed day (holiday, renovation) won't drag the weekday average down.",
                                    "Een gesloten dag (vakantie, verbouwing) trekt het weekdaggemiddelde dan "
                                    "niet omlaag."))

    history_days = (until - res.visit_date.min()).days
    if history_days < 21:
        when = f"{until.day} {T.month_short(until)} {until.year}"
        st.error(tr(f"Your file only goes back {history_days} days before {when}. "
                    "The forecast needs at least 3 weeks of past visits, ideally 3 months.",
                    f"Je bestand gaat maar {history_days} dagen terug vóór {when}. De voorspelling "
                    "heeft minstens 3 weken aan eerdere bezoeken nodig, liefst 3 maanden."))
        return

    with st.spinner(tr("Checking the forecast on your own last 8 weeks…",
                       "De voorspelling wordt gecontroleerd op je eigen laatste 8 weken…")):
        p, check = run_plan(res, until, None if actual is None else actual, days, skip_closed)
    if source == "sample":
        p["actual"] = p.date.map(sample_actual())
    if walk_in:
        p = P.add_walk_ins(p, *walk_in)
    has_walk_ins = "walk_ins" in p

    notes = [tr(f"{len(res):,} reservations read", f"{len(res):,} reserveringen gelezen".replace(",", "."))]
    if has_walk_ins:
        notes.append(tr(f"walk-ins added: {walk_in[0]:.0%} of guests Fri-Sun, {walk_in[1]:.0%} Mon-Thu",
                        f"walk-ins opgeteld: {walk_in[0]:.0%} van de gasten vr-zo, {walk_in[1]:.0%} ma-do"))
    if prep.dropped:
        notes.append(tr(f"{prep.dropped:,} rows skipped (no readable date or no guests)",
                        f"{prep.dropped} rijen overgeslagen (geen leesbare datum of geen gasten)"))
    st.caption(" · ".join(notes + [T.message(n) for n in prep.notes]))

    open_days = p[~p.closed]
    if open_days.empty:
        st.warning(tr("No open days found to forecast. Try switching off 'Skip days with no guests'.",
                      "Geen open dagen gevonden om te voorspellen. Zet 'Dagen zonder gasten overslaan' uit."))
        return

    # ---- headline figures
    has_actual = "actual" in p
    busiest = open_days.loc[open_days.forecast.idxmax()]
    status = p.vs_usual_pct.map(lambda v: ui.day_status(v, UNUSUAL))
    unusual = open_days[status[open_days.index] != "usual"]
    total, usual_total = open_days.forecast.sum(), open_days.usual.sum()
    change = total / usual_total - 1 if usual_total else 0.0
    if abs(change) < 0.005:
        against_usual = tr(f"about the same as a usual {days} days",
                           f"ongeveer gelijk aan gewone {days} dagen")
    elif change > 0:
        against_usual = tr(f"{change:.0%} more than a usual {days} days",
                           f"{change:.0%} meer dan gewone {days} dagen")
    else:
        against_usual = tr(f"{-change:.0%} fewer than a usual {days} days",
                           f"{-change:.0%} minder dan gewone {days} dagen")
    one = len(unusual) == 1
    total_text = f"{total:,.0f}" if T.lang() == "en" else f"{total:,.0f}".replace(",", ".")
    figures = [
        (total_text, tr(f"guests expected in the next {days} days",
                        f"gasten verwacht in de komende {days} dagen"), against_usual),
        (day_label(busiest.date), tr("busiest day", "drukste dag"),
         tr(f"{busiest.forecast:.0f} guests", f"{busiest.forecast:.0f} gasten")),
        (tr(f"{len(unusual)} day{'' if one else 's'}", f"{len(unusual)} {'dag' if one else 'dagen'}"),
         tr("needs a second look" if one else "need a second look", "verdient een tweede blik" if one
            else "verdienen een tweede blik"),
         tr(f"{UNUSUAL}% or more away from the usual weekday",
            f"{UNUSUAL}% of meer naast de gewone weekdag")),
    ]
    if len(check):
        row7 = check[check.days_ahead == min(7, check.days_ahead.max())].iloc[0]
        figures.append((tr(f"± {row7.pace_error:.0f} guests", f"± {row7.pace_error:.0f} gasten"),
                        tr(f"average miss on your {'reservations, ' if has_walk_ins else ''}last 8 weeks, "
                           f"{int(row7.days_ahead)} days ahead",
                           f"gemiddelde afwijking op je {'reserveringen, ' if has_walk_ins else ''}laatste "
                           f"8 weken, {int(row7.days_ahead)} dagen vooruit"),
                        tr(f"the weekday average alone: ± {row7.usual_error:.0f}",
                           f"alleen het weekdaggemiddelde: ± {row7.usual_error:.0f}")))
    st.html(ui.stats(figures))

    # ---- the days ahead as a calendar
    tiles = []
    for i, r in p.iterrows():
        tile = {"date": r.date, "closed": bool(r.closed)}
        if not r.closed:
            likely = "" if np.isnan(r.low) else tr(f"likely {r.low:.0f} to {r.high:.0f}",
                                                   f"waarschijnlijk {r.low:.0f} tot {r.high:.0f}")
            came_in = has_actual and not np.isnan(r.actual)
            came = tr(f"{r.actual:.0f} came", f"{r.actual:.0f} kwamen") if came_in else ""
            title = [tr(f"Forecast: {r.forecast:.0f} guests", f"Voorspelling: {r.forecast:.0f} gasten"),
                     likely.capitalize(),
                     tr(f"Usual for a {T.day_name(r.date)}: {r.usual:.0f}",
                        f"Gewoonlijk op {T.day_name(r.date)}: {r.usual:.0f}"),
                     tr(f"Booked so far: {r.booked:.0f}", f"Tot nu toe geboekt: {r.booked:.0f}"),
                     tr(f"{r.actual:.0f} actually came", f"Er kwamen er echt {r.actual:.0f}") if came_in else ""]
            if has_walk_ins:
                title.insert(1, tr(f"{r.reserved:.0f} from reservations, about {r.walk_ins:.0f} walk-ins",
                                   f"{r.reserved:.0f} uit reserveringen, ongeveer {r.walk_ins:.0f} walk-ins"))
            tile.update(forecast=r.forecast, usual=r.usual, status=status[i],
                        vs_usual_pct=r.vs_usual_pct, sub=came or likely,
                        title=". ".join(t for t in title if t) + ".")
        tiles.append(tile)
    st.html(ui.calendar(tiles))
    st.html(ui.key([
        ("bar", ui.CALM, tr("about usual", "ongeveer gewoon")),
        ("bar", ui.HOT, tr(f"busier than usual by {UNUSUAL}% or more", f"{UNUSUAL}% of meer drukker dan gewoon")),
        ("bar", ui.COOL, tr(f"quieter by {UNUSUAL}% or more", f"{UNUSUAL}% of meer rustiger")),
        ("tick", ui.INK, tr("line across each bar: the usual for that weekday",
                            "streep door elke balk: het gewone aantal voor die weekdag")),
    ]))

    # ---- chart: how sure each day is
    st.subheader(tr("Forecast and likely range", "Voorspelling en waarschijnlijke marge"))
    c = open_days.copy()
    c["day"] = c.date.map(day_label)
    c["label"] = c.date.map(lambda d: T.short_date(d).replace(" ", "|", 1))  # two lines on the axis
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
    tooltip = [alt.Tooltip("day:N", title=tr("Day", "Dag")),
               alt.Tooltip("forecast:Q", title=tr("Forecast", "Voorspelling"), format=".0f"),
               alt.Tooltip("low:Q", title=tr("Likely from", "Waarschijnlijk vanaf"), format=".0f"),
               alt.Tooltip("high:Q", title=tr("Likely to", "Waarschijnlijk tot"), format=".0f"),
               alt.Tooltip("usual:Q", title=tr("Usual for this weekday", "Gewoon voor deze weekdag"), format=".0f"),
               alt.Tooltip("booked:Q", title=tr("Booked so far", "Tot nu toe geboekt"))]
    if has_walk_ins:
        tooltip[2:2] = [alt.Tooltip("reserved:Q", title=tr("From reservations", "Uit reserveringen"), format=".0f"),
                        alt.Tooltip("walk_ins:Q", title=tr("Walk-ins (estimate)", "Walk-ins (schatting)"), format=".0f")]
    if has_actual:
        tooltip.append(alt.Tooltip("actual:Q", title=tr("Actually came", "Echt gekomen")))
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
    marks = [("bar", ui.CALM, tr("forecast", "voorspelling")),
             ("whisker", ui.INK, tr("likely range (8 in 10 days)", "waarschijnlijke marge (8 op de 10 dagen)")),
             ("tick", ui.INK, tr("usual for that weekday", "gewoon voor die weekdag")),
             ("ring", ui.INK, tr("booked so far", "tot nu toe geboekt"))]
    if has_actual:
        marks.append(("diamond", ui.INK, tr("what really happened", "wat er echt gebeurde")))
    st.html(ui.key(marks))
    st.altair_chart(ui.polish(alt.layer(*layers), height=330), width="stretch")

    # ---- staffing + table
    st.subheader(tr("Staffing plan", "Personeelsplanning"))
    with st.expander(tr("Staffing rules: change these to your own", "Personeelsregels: pas ze aan naar je eigen zaak"),
                     expanded=False):
        r1, r2 = st.columns([2, 1])
        # Each language has its own copy of the rules, so the default role names read naturally.
        default_rules = P.DEFAULT_RULES if T.lang() == "en" else P.DEFAULT_RULES.assign(
            role=P.DEFAULT_RULES.role.map({"Floor": "Bediening", "Bar": "Bar", "Kitchen": "Keuken"}))
        rules = r1.data_editor(
            default_rules, num_rows="dynamic", hide_index=True, key=f"rules_{T.lang()}", width="stretch",
            column_config={
                "role": st.column_config.TextColumn(tr("Role", "Functie")),
                "guests_per_person": st.column_config.NumberColumn(tr("Guests per person", "Gasten per persoon"),
                                                                   min_value=1, step=1),
                "minimum": st.column_config.NumberColumn(tr("Minimum on an open day", "Minimum op een open dag"),
                                                         min_value=0, step=1),
            })
        r2.caption(tr("People = guests ÷ guests per person, rounded up, never below the minimum. "
                      "Add a row for every role you schedule.",
                      "Mensen = gasten ÷ gasten per persoon, naar boven afgerond, nooit onder het minimum. "
                      "Voeg een rij toe voor elke functie die je inroostert."))
        staff_options = {"forecast": tr("The forecast", "De voorspelling"),
                         "high": tr("The top of the likely range", "De bovenkant van de marge")}
        staff_on = r2.radio(tr("Staff for", "Plan personeel voor"), list(staff_options),
                            format_func=staff_options.get, key="staff_on",
                            help=tr("Plan for the top of the range when running short costs more than one "
                                    "extra person.",
                                    "Plan voor de bovenkant als te weinig mensen je meer kost dan één "
                                    "persoon extra."))
    staff = P.staffing(p.assign(high=p.high.fillna(p.forecast)), rules, use=staff_on)

    words = {"busier": tr("busier", "drukker"), "quieter": tr("quieter", "rustiger")}

    def flag(r):
        if r.closed:
            return tr("Closed", "Gesloten")
        s = ui.day_status(r.vs_usual_pct, UNUSUAL)
        if s != "usual":
            return f"{'↑' if s == 'busier' else '↓'} {abs(r.vs_usual_pct):.0f}% {words[s]}"
        return ""

    forecast_col, vs_usual_col = tr("Forecast", "Voorspelling"), tr("vs usual", "t.o.v. gewoon")
    table = pd.DataFrame({
        tr("Date", "Datum"): p.date.map(day_label),
        tr("Booked so far", "Tot nu toe geboekt"): p.booked,
        tr("Usual", "Gewoon"): p.usual.round(),
        **({tr("From reservations", "Uit reserveringen"): p.reserved.round(),
            tr("Walk-ins (estimate)", "Walk-ins (schatting)"): p.walk_ins.round()}
           if has_walk_ins else {}),
        forecast_col: p.forecast.round(),
        tr("Likely range", "Waarschijnlijke marge"): [("" if np.isnan(lo) else f"{lo:.0f}-{hi:.0f}")
                                                      for lo, hi in zip(p.low, p.high)],
        vs_usual_col: p.apply(flag, axis=1),
    })
    if has_actual:
        table[tr("Actually came", "Echt gekomen")] = p.actual
    for col in staff.columns[1:]:
        table[col] = staff[col]

    def flag_colour(v: str) -> str:
        return ("background-color: #FCE4DA" if v.startswith("↑")
                else "background-color: #DEEAF7" if v.startswith("↓") else "")

    numbers = [col for col in table.columns if table[col].dtype.kind in "if"]
    shown = table.style.map(flag_colour, subset=[vs_usual_col]).format("{:.0f}", subset=numbers, na_rep="")
    st.dataframe(shown, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
                 column_config={
                     forecast_col: st.column_config.ProgressColumn(
                         format="%d", min_value=0, max_value=float(np.nanmax([p.forecast.max(), 1])),
                         color=ui.CALM, width="medium"),
                     vs_usual_col: st.column_config.TextColumn(width="medium"),
                 })
    with st.container(horizontal=True):
        st.download_button(tr("Download Excel", "Download Excel"), to_excel(table), f"guest_plan_{until:%Y%m%d}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/download:")
        st.download_button(tr("Download CSV", "Download CSV"), table.to_csv(index=False).encode(),
                           f"guest_plan_{until:%Y%m%d}.csv", "text/csv", icon=":material/download:")

    if has_actual:
        hit = open_days.assign(actual=p.actual).dropna(subset=["actual"])
        mae_p = (hit.forecast - hit.actual).abs().mean()
        mae_u = (hit.usual - hit.actual).abs().mean()
        inside = ((hit.actual >= hit.low) & (hit.actual <= hit.high)).mean()
        st.success(tr(
            f"**How this sample turned out:** the forecast was off by **{mae_p:.0f}** guests a day "
            f"on average (the usual weekday number: {mae_u:.0f}), and **{inside:.0%}** of these days "
            "landed inside the likely range. Over 50 different two-week plans on this data "
            "(every other week, Sep 2016 - Aug 2017, both hotels), 79-80% of days landed inside it, "
            "so a single fortnight can come out better or worse.",
            f"**Hoe dit voorbeeld uitpakte:** de voorspelling zat gemiddeld **{mae_p:.0f}** gasten per "
            f"dag ernaast (het gewone weekdaggetal: {mae_u:.0f}), en **{inside:.0%}** van deze dagen viel "
            "binnen de waarschijnlijke marge. Over 50 verschillende tweeweekse plannen op deze data "
            "(om de week, sep 2016 - aug 2017, beide hotels) viel 79-80% van de dagen erbinnen, dus "
            "één losse twee weken kan beter of slechter uitvallen."))

    with st.expander(tr("How sure is it, on your own data?", "Hoe zeker is het, op je eigen data?")):
        if len(check):
            ch = check.rename(columns={"days_ahead": tr("Days ahead", "Dagen vooruit"),
                                       "days_scored": tr("Days checked", "Dagen gecontroleerd"),
                                       "pace_error": tr("Forecast off by (avg)", "Voorspelling ernaast (gem.)"),
                                       "usual_error": tr("Usual weekday off by (avg)", "Gewone weekdag ernaast (gem.)"),
                                       "pace_p80": tr("8 in 10 days within", "8 op de 10 dagen binnen")}).round(1)
            st.dataframe(ch, hide_index=True)
            st.caption(tr("Replayed on your last 8 weeks: for each past day the reservation book is rebuilt "
                          "as it stood back then, so the check never uses information it couldn't have had.",
                          "Nagespeeld op je laatste 8 weken: voor elke dag wordt het reserveringsboek "
                          "herbouwd zoals het er toen voor stond, dus de controle gebruikt nooit informatie "
                          "die toen nog niet bestond."))
            if has_walk_ins:
                st.caption(tr("This checks the reservations part only. The walk-in estimate can't be "
                              "checked without real guest totals: upload them from your till to replace it.",
                              "Dit controleert alleen het reserveringsdeel. De walk-in-schatting kan niet "
                              "gecontroleerd worden zonder echte gastentotalen: upload ze uit je kassa om "
                              "de schatting te vervangen."))
        else:
            st.write(tr("Not enough history yet to check (needs about 10 weeks).",
                        "Nog niet genoeg geschiedenis om te controleren (daar zijn ongeveer 10 weken voor nodig)."))


with plan_tab:
    plan_page()


# ---------------------------------------------------------------- proof tab -----------------

@st.cache_resource(show_spinner=False)  # the callers draw the spinner, in the visitor's language
def hotels():
    return load_hotels()


@st.cache_data(show_spinner=False)
def backtest():
    h = hotels()
    model = ml.train(build_rows(h, *TRAIN, horizons=range(1, 31)))
    test = build_rows(h, *TEST, horizons=range(1, 31))
    test["gradient_boosting"] = ml.predict(model, test)
    return test


SERIES = {  # column -> (name, colour, line width, dash, glyph in the key)
    "weekday_average": (tr("Weekday average", "Weekdaggemiddelde"), ui.BASELINE, 1.75, [5, 3], "dash"),
    "gradient_boosting": ("Gradient boosting (ML)", ui.ML, 1.75, [1, 0], "thin"),
    "pace": (tr("Booking pace (rule)", "Boekingstempo (regel)"), ui.PACE, 2.75, [1, 0], "line"),
    "actual": (tr("Actual guests", "Echte gasten"), ui.INK, 2.75, [1, 0], "line"),
}
COLUMNS = {column: name for column, (name, *_) in SERIES.items()}


def against_baseline(pct: float) -> str:
    if pct >= 0:
        return tr(f"{pct:.0f}% lower than the weekday average", f"{pct:.0f}% lager dan het weekdaggemiddelde")
    return tr(f"{-pct:.0f}% higher than the weekday average", f"{-pct:.0f}% hoger dan het weekdaggemiddelde")


with proof_tab:
    st.markdown(tr(
        "Tested on **119,390 real hotel bookings** (two hotels in Portugal, 2015-2017), scored on six "
        "months the models never saw. Every forecast uses only bookings that existed on the morning "
        "it was made.",
        "Getest op **119.390 echte hotelboekingen** (twee hotels in Portugal, 2015-2017), beoordeeld op "
        "zes maanden die de modellen nooit gezien hebben. Elke voorspelling gebruikt alleen boekingen "
        "die bestonden op de ochtend dat ze gemaakt werd."
    ))
    with st.spinner(tr("Replaying six months of forecasts…", "Zes maanden aan voorspellingen naspelen…")):
        test = backtest()
    c1, c2, c3 = st.columns([1.2, 1, 1])
    hotel_names = {"City Hotel": tr("City Hotel", "Stadshotel"), "Resort Hotel": tr("Resort Hotel", "Resorthotel")}
    hotel = c1.segmented_control("Hotel", list(hotel_names), format_func=hotel_names.get,
                                 default="City Hotel", required=True, key="proof_hotel")
    horizon = c2.slider(tr("Forecast made … days ahead", "Voorspelling … dagen vooruit gemaakt"), 1, 30, 7,
                        key="proof_horizon")
    month_starts = list(pd.date_range(*TEST, freq="MS"))
    months = c3.select_slider(tr("Months shown", "Getoonde maanden"), options=month_starts,
                              format_func=T.month_year, key="proof_months",
                              value=(pd.Timestamp("2017-05-01"), pd.Timestamp("2017-06-01")))
    g = test[(test.hotel == hotel) & (test.horizon == horizon)].copy()
    s = summarise(g).iloc[0]

    st.subheader(tr("How far off, in guests per day", "Hoeveel ernaast, in gasten per dag"))
    st.html(ui.error_bars([
        (COLUMNS["weekday_average"], s.weekday_average, ui.BASELINE,
         tr("what most managers do by feel", "wat de meeste managers op gevoel doen")),
        (COLUMNS["pace"], s.pace, ui.PACE, against_baseline(s.pace_vs_weekday_pct)),
        (COLUMNS["gradient_boosting"], s.gradient_boosting, ui.ML, against_baseline(s.ml_vs_weekday_pct)),
    ]))
    st.caption(tr(f"Mean absolute error over every test day, 1 Mar - 31 Aug 2017. Lower is better. "
                  f"This hotel averages {s.average_guests:.0f} guests arriving per day.",
                  f"Gemiddelde absolute fout over elke testdag, 1 mrt - 31 aug 2017. Lager is beter. "
                  f"Dit hotel krijgt gemiddeld {s.average_guests:.0f} aankomende gasten per dag."))

    start = pd.Timestamp(months[0])
    end = pd.Timestamp(months[1]) + pd.offsets.MonthEnd(0)
    window = g[g.day.between(start, end)]
    long = window.melt(id_vars="day", value_vars=list(COLUMNS), var_name="series", value_name="guests")
    long["series"] = long.series.map(COLUMNS)
    long["guests"] = long.guests.round(0)
    lines = [  # drawn in this order, so the two that matter most end up on top
        alt.Chart(long[long.series == name]).mark_line(
            color=colour, strokeWidth=width, strokeDash=dash, strokeJoin="round").encode(
            x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=10)),
            y=alt.Y("guests:Q", title=None),
            tooltip=[alt.Tooltip("day:T", title=tr("Day", "Dag"), format="%a %d %b %Y"),
                     alt.Tooltip("series:N", title=tr("Series", "Lijn")),
                     alt.Tooltip("guests:Q", title=tr("Guests", "Gasten"))])
        for name, colour, width, dash, _ in SERIES.values()
    ]
    st.subheader(tr("Forecast against what really happened", "Voorspelling tegenover wat er echt gebeurde"))
    st.html(ui.key([(glyph, colour, name) for name, colour, _, _, glyph in reversed(SERIES.values())]))
    st.altair_chart(ui.polish(alt.layer(*lines).interactive(bind_y=False), height=340), width="stretch")

    st.subheader(tr("Look inside one day", "Kijk in één dag"))
    g["miss"] = (g.weekday_average - g.actual).abs() - (g.pace - g.actual).abs()
    default_day = g.sort_values("miss").day.iloc[-1].date()
    day = pd.Timestamp(st.date_input(tr("Day", "Dag"), value=default_day, min_value=TEST[0].date(),
                                     max_value=TEST[1].date(), width=260))
    row = g[g.day == day].iloc[0]
    h = hotels()[hotel]
    refs = reference_days(h, day, horizon)
    ks = np.arange(60, -1, -1)
    this_day = T.short_date(day)
    normal_day = tr(f"A normal {T.day_name(day)}", f"Een normale {T.day_name(day)}")
    curve = pd.DataFrame({
        "days_before": -ks,
        this_day: [h.booked(day, k) for k in ks],
        normal_day: [np.mean([h.booked(r, k) for r in refs]) for k in ks],
    }).melt(id_vars="days_before", var_name="series", value_name="guests")
    # Same colours as the plan: orange-red when the book runs ahead of normal, blue when behind.
    ahead = row.booked_now - row.booked_normally
    this_colour = ui.HOT if ahead > 0 else ui.COOL if ahead < 0 else ui.CALM
    left, right = st.columns([2, 1])
    left.html(ui.key([("line", this_colour, this_day), ("thin", ui.BASELINE, normal_day),
                      ("dash", ui.INK2, tr("the day the forecast was made", "de dag van de voorspelling"))]))
    tip = [alt.Tooltip("series:N", title=tr("Series", "Lijn")),
           alt.Tooltip("days_before:Q", title=tr("days before arrival", "dagen voor aankomst")),
           alt.Tooltip("guests:Q", title=tr("guests in the book", "gasten in het boek"), format=".0f")]
    axes = dict(x=alt.X("days_before:Q", title=tr("Days before arrival", "Dagen voor aankomst"),
                        axis=alt.Axis(tickCount=12)),
                y=alt.Y("guests:Q", title=None))
    one_day = alt.layer(
        alt.Chart(curve[curve.series == normal_day]).mark_line(color=ui.BASELINE, strokeWidth=2).encode(tooltip=tip, **axes),
        alt.Chart(curve[curve.series == this_day]).mark_line(color=this_colour, strokeWidth=3,
                                                              strokeJoin="round").encode(tooltip=tip, **axes),
        alt.Chart(pd.DataFrame({"x": [-(horizon - 1)]})).mark_rule(strokeDash=[4, 3], color=ui.INK2).encode(x="x:Q"),
    )
    left.altair_chart(ui.polish(one_day, height=300), width="stretch")
    s_ = "s" if horizon > 1 else ""
    long_day = T.long_date(day, year=False)
    weekdays = tr(f"{T.day_name(day)}s", f"{T.day_name(day)}en")
    right.markdown(tr(
        f"""
**Standing {horizon} day{s_} before {long_day}:**

1. The last {len(refs)} {weekdays} averaged **{row.weekday_average:.0f}** guests.
2. The book held **{row.booked_now}** guests; on a normal {T.day_name(day)} it held
   **{row.booked_normally:.0f}** at this point.
3. So bookings are running **{row.booked_now - row.booked_normally:+.0f}** against normal,
   and the pace forecast says **{row.pace:.0f}** (the correction is capped at ±50%).

The ML model said **{row.gradient_boosting:.0f}**. **{row.actual}** guests arrived.
""",
        f"""
**{horizon} {'dagen' if horizon > 1 else 'dag'} voor {long_day}:**

1. De laatste {len(refs)} {weekdays} kwamen gemiddeld uit op **{row.weekday_average:.0f}** gasten.
2. Het boek had **{row.booked_now}** gasten; op een normale {T.day_name(day)} waren dat er op dit
   moment **{row.booked_normally:.0f}**.
3. De boekingen lopen dus **{row.booked_now - row.booked_normally:+.0f}** ten opzichte van normaal,
   en de tempovoorspelling zegt **{row.pace:.0f}** (de correctie is begrensd op ±50%).

Het ML-model zei **{row.gradient_boosting:.0f}**. Er kwamen **{row.actual}** gasten.
"""))

    with st.expander(tr("Results at every horizon", "Resultaten voor elke termijn")):
        full = summarise(test[test.horizon.isin([1, 3, 7, 14, 30])])
        full = full.rename(columns={"horizon": tr("days ahead", "dagen vooruit"),
                                    "average_guests": tr("avg guests", "gem. gasten"),
                                    **{k: v for k, v in COLUMNS.items() if k != "actual"},
                                    "same_day_last_year": tr("Same day last year", "Zelfde dag vorig jaar")})
        st.dataframe(full.drop(columns=["days", "pace_vs_weekday_pct", "ml_vs_weekday_pct"]).round(1),
                     hide_index=True, width="stretch")

    with st.expander(tr("How it works, and what it doesn't show", "Hoe het werkt, en wat het niet laat zien")):
        repo_link = f"[{REPO.split('github.com/')[1]}]({REPO})"
        st.markdown(tr(
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

Code, tests and charts: {repo_link}
""",
            f"""
- **Geen blik in de toekomst.** Elke voorspelling gebruikt alleen boekingen die op de ochtend van de
  voorspelling in het boek stonden, en alleen dagen die al voorbij waren. Het ML-model leert op
  jul 2016 - feb 2017 en wordt beoordeeld op mrt - aug 2017. Tests in de repo controleren allebei.
- **Eenvoudig wint van slim tot een week vooruit.** De regel van twee regels code verslaat gradient
  boosting op elke termijn van 1 tot 7 dagen; ML loopt pas een maand vooruit voor, als het boek nog
  dun is.
- **Hotels boeken verder vooruit dan restaurants.** De winst op 1 dag is groter dan een restaurant
  met walk-ins zou zien; de cijfers voor 7 en 14 dagen zijn de eerlijkere vergelijking.
- **Data:** Antonio, de Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22,
  41-49, CC BY 4.0. De auteurs hebben hotel- en gastgegevens geanonimiseerd.

Code, tests en grafieken: {repo_link}
"""))

@st.cache_resource(show_spinner=False)
def sample_backend():
    """The assistant's tools on the sample export: the same numbers as the Plan tab's sample."""
    from agent.real_backend import RealBackend
    return RealBackend()


with ask_tab:
    from app_pages.assistant import render as render_assistant

    st.info(tr("The assistant answers from the **sample hotel's** forecast (export of 31 May 2017, so "
               "**today is Thursday 1 June 2017**) with the default staffing rules. Every number it "
               "gives comes from the same tools as the Plan tab; it never sees your uploaded file.",
               "De assistent antwoordt vanuit de voorspelling van het **voorbeeldhotel** (export van "
               "31 mei 2017, dus **vandaag is donderdag 1 juni 2017**) met de standaard personeelsregels. "
               "Elk getal komt uit dezelfde hulpmiddelen als het tabblad Plan; je geüploade bestand "
               "ziet de assistent nooit."),
            icon=":material/info:")
    render_assistant(sample_backend())

st.html(ui.footer(REPO))
