"""The app's look: colours, the page styles, and the HTML building blocks.

Nothing here computes a forecast. Each function takes numbers the app already has and returns
HTML for ``st.html``; ``polish`` styles an Altair chart. Colours match .streamlit/config.toml.

Colour carries one meaning everywhere: orange-red is busier than usual, blue is quieter,
dark green is about usual. Text never relies on colour alone: flagged days also carry an
arrow and a percentage.
"""
from __future__ import annotations

from base64 import b64encode
from html import escape

import streamlit as st

from app_pages.i18n import short_date, tr, vega_locale

# ---------------------------------------------------------------- colours -------------------

BAND, PAPER = "#0B2E28", "#F2F4EC"             # the dark header and the text on it
INK, INK2, MUTED, LINE = "#10201D", "#465551", "#64716D", "#DCE2DE"
PAGE, CARD = "#F4F6F3", "#FFFFFF"
CALM, HOT, COOL = "#2F5F55", "#E8552F", "#3B82C4"   # about usual, busier, quieter
PACE, ML, BASELINE = "#12957F", "#7B4FD6", "#8A9692"  # the three forecasts on the accuracy tab
STATUS = {"usual": CALM, "busier": HOT, "quieter": COOL}
BODY_FONT = "Instrument Sans, system-ui, sans-serif"

# The header's example: guests in the City Hotel's book for Friday 28 April 2017 and for a
# normal Friday (the 6 Fridays before), from 60 days before arrival to the day itself.
# These are real numbers from the benchmark data; tests/test_ui.py recomputes them.
STORY_DAY = "2017-04-28"
STORY_THIS = [210, 210, 210, 210, 210, 210, 210, 210, 219, 219, 219, 217, 219, 219, 219, 220,
              220, 220, 220, 220, 220, 218, 196, 196, 192, 191, 192, 192, 192, 192, 193, 193,
              210, 218, 228, 236, 235, 235, 239, 236, 235, 241, 243, 218, 232, 228, 228, 230,
              230, 233, 233, 233, 242, 246, 244, 244, 239, 239, 239, 243, 241]
STORY_NORMAL = [137.3, 138.5, 137.7, 138.8, 141.2, 144.3, 145.7, 145.2, 144.8, 146.2, 147.8,
                147.7, 148.3, 148.7, 149.5, 149.5, 150.7, 150.0, 150.7, 149.5, 150.8, 150.3,
                152.8, 152.3, 152.0, 152.3, 152.3, 153.8, 153.2, 153.2, 156.2, 156.7, 156.0,
                154.2, 152.5, 154.3, 155.8, 156.0, 154.3, 153.3, 151.7, 150.3, 150.0, 150.0,
                149.3, 153.2, 153.3, 154.3, 152.8, 153.3, 153.8, 152.3, 152.2, 153.8, 156.0,
                155.3, 155.8, 157.3, 159.3, 159.8, 161.8]
STORY = {"booked": 244, "normally": 156, "weekday_average": 162, "pace": 243, "actual": 241}

# ---------------------------------------------------------------- page styles ---------------

CSS = f"""
<style>
:root {{
  --gf-band:{BAND}; --gf-paper:{PAPER}; --gf-band-soft:#C9D9D2; --gf-band-line:#3C6A5F;
  --gf-ink:{INK}; --gf-ink2:{INK2}; --gf-muted:{MUTED}; --gf-line:{LINE};
  --gf-page:{PAGE}; --gf-card:{CARD};
  --gf-calm:{CALM}; --gf-hot:{HOT}; --gf-cool:{COOL};
  --gf-hot-bg:#FCE4DA; --gf-cool-bg:#DEEAF7;
  --gf-display:"Bricolage Grotesque","Instrument Sans",system-ui,sans-serif;
}}

/* Streamlit's top bar joins the dark band, so its buttons stay readable while scrolling. */
[data-testid="stHeader"] {{ background:var(--gf-band); height:2.75rem; min-height:2.75rem; }}
[data-testid="stToolbar"], [data-testid="stToolbar"] button, [data-testid="stToolbar"] a,
[data-testid="stToolbar"] span, [data-testid="stStatusWidget"] {{ color:var(--gf-paper); }}
[data-testid="stToolbar"] svg {{ fill:currentColor; }}

h2, h3 {{ font-variation-settings:"wdth" 86; letter-spacing:-0.005em; }}

/* ---- header band: full width, whatever the page padding is */
.gf-hero {{
  background:var(--gf-band); color:var(--gf-paper);
  box-shadow:0 0 0 100vmax var(--gf-band); clip-path:inset(-100vmax -100vmax 0 -100vmax);
  padding:4rem 0 5.4rem;
  display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr); gap:2rem 3.5rem; align-items:end;
}}
.gf-hero h1 {{
  font-family:var(--gf-display); font-weight:800; font-variation-settings:"wdth" 75, "opsz" 96;
  font-size:clamp(2.7rem, 5.6vw, 4.9rem); line-height:.92; letter-spacing:-0.015em;
  color:var(--gf-paper); margin:0 0 1.2rem; padding:0; text-wrap:balance;
}}
.gf-hero p {{ margin:0; }}
.gf-lede {{ font-size:1.12rem; line-height:1.5; max-width:34em; color:var(--gf-band-soft); }}
.gf-hero a {{ color:var(--gf-paper); text-underline-offset:3px; }}
.gf-hero a:hover {{ color:#FFB59C; }}
.gf-hero a:focus-visible {{ outline:2px solid #FFB59C; outline-offset:3px; border-radius:2px; }}
.gf-cta {{ margin-top:1.3rem !important; }}
.gf-hero a.gf-code {{
  display:inline-block; padding:.6rem 1.25rem; border-radius:999px; background:var(--gf-paper);
  color:var(--gf-band); font-weight:650; line-height:1.2; text-decoration:none;
}}
.gf-hero a.gf-code:hover {{ background:#FFB59C; color:var(--gf-band); }}
.gf-hero a.gf-code:focus-visible {{ border-radius:999px; }}
.gf-story {{ margin:0; }}
.gf-plot {{ position:relative; max-width:35rem; container-type:inline-size; }}
.gf-plot img {{ width:100%; height:auto; display:block; }}
.gf-plot span {{ position:absolute; white-space:nowrap; line-height:1; font-size:clamp(.68rem, 2.6cqi, .8rem); }}
.gf-tick {{ color:#8FB3A6; }}
.gf-lab {{ font-weight:600; transform:translateY(-50%); }}
.gf-gap {{ font-family:var(--gf-display); font-size:clamp(.78rem, 3.2cqi, 1rem) !important; font-weight:700; color:var(--gf-paper); transform:translate(-100%, -50%); }}
.gf-late {{ opacity:0; animation:gf-show .5s 1.6s ease-out forwards; }}
.gf-caption {{ font-size:.95rem; line-height:1.45; color:var(--gf-band-soft); margin-top:.6rem !important; max-width:38em; }}
.gf-caption b {{ color:var(--gf-paper); font-weight:600; }}
@keyframes gf-show {{ to {{ opacity:1; }} }}
@media (prefers-reduced-motion: reduce) {{
  .gf-late {{ animation:none; opacity:1; }}
}}
@media (max-width: 900px) {{
  .gf-hero {{ grid-template-columns:minmax(0,1fr); padding-top:5.6rem; }}  /* room for the language switch */
}}

/* ---- the English / Nederlands switch, top right on the dark band */
.st-key-lang_switch {{ position:relative; z-index:2; margin-top:3.6rem; margin-bottom:-6.6rem; }}
.st-key-lang_switch button {{
  background:transparent; color:var(--gf-band-soft); border-color:var(--gf-band-line);
  min-height:2rem; padding:.2rem .85rem;
}}
.st-key-lang_switch button:hover {{ color:var(--gf-paper); border-color:var(--gf-band-soft); }}
.st-key-lang_switch button[aria-checked="true"] {{
  background:var(--gf-paper) !important; color:var(--gf-band) !important; border-color:var(--gf-paper) !important;
}}
.st-key-lang_switch button p {{ font-size:.85rem; font-weight:600; }}

/* ---- tabs sit on the bottom edge of the band, like index tabs on a book */
[data-testid="stTabs"] {{ position:relative; z-index:1; margin-top:-4.1rem; }}
[data-testid="stTabs"] [role="tablist"] {{ gap:.4rem; border:0; box-shadow:none; }}
[data-testid="stTabs"] [role="tablist"]::before, [data-testid="stTabs"] [role="tablist"]::after {{ display:none; }}
[data-testid="stTab"] {{
  height:3.1rem; padding:0 1.35rem; margin:0; border-radius:.75rem .75rem 0 0; border:0;
  background:rgba(242,244,236,.09); color:var(--gf-band-soft);
}}
[data-testid="stTab"] p {{ font-size:1rem; font-weight:600; color:inherit; }}
[data-testid="stTab"]:hover {{ background:rgba(242,244,236,.17); color:var(--gf-paper); }}
[data-testid="stTab"][aria-selected="true"] {{ background:var(--gf-page); color:var(--gf-ink); }}
[data-testid="stTab"]:focus-visible {{ outline:2px solid #FFB59C; outline-offset:2px; }}
[data-testid="stTab"] .react-aria-SelectionIndicator {{ display:none; }}
[data-testid="stTabPanel"] {{ padding-top:1.5rem; }}
/* the arrows that appear when the tabs don't fit (phones) */
[data-testid="stTabsScrollLeft"], [data-testid="stTabsScrollRight"] {{ color:var(--gf-paper); height:3.1rem; width:2.1rem; }}
[data-testid="stTabsScrollRight"] {{ background-image:linear-gradient(to right, rgba(11,46,40,0), var(--gf-band) 45%); }}
[data-testid="stTabsScrollLeft"] {{ background-image:linear-gradient(to left, rgba(11,46,40,0), var(--gf-band) 45%); }}
@media (max-width: 640px) {{
  [data-testid="stTab"] {{ padding:0 .95rem; }}
  [data-testid="stTab"] p {{ font-size:.94rem; }}
}}

/* ---- the headline figures */
.gf-stats {{ display:grid; grid-template-columns:1.3fr 1fr 1fr 1fr; margin:.6rem 0 .4rem; container-type:inline-size; }}
.gf-stat {{ padding:.1rem 1.6rem; border-left:1px solid var(--gf-line); min-width:0; }}
.gf-stat:first-child {{ padding-left:0; border-left:0; }}
.gf-v {{ font-family:var(--gf-display); font-weight:700; font-variation-settings:"wdth" 82; font-size:2.1rem; line-height:1.05; letter-spacing:-0.01em; color:var(--gf-ink); }}
.gf-lead .gf-v {{ font-size:3.7rem; font-weight:800; font-variation-settings:"wdth" 75, "opsz" 96; line-height:.95; }}
.gf-l {{ color:var(--gf-ink2); font-size:.95rem; line-height:1.35; margin-top:.4rem; }}
.gf-d {{ color:var(--gf-muted); font-size:.875rem; line-height:1.35; }}
@media (max-width: 900px) {{
  .gf-stats {{ grid-template-columns:1fr 1fr; row-gap:1.3rem; }}
  .gf-stat:nth-child(odd) {{ padding-left:0; border-left:0; }}
}}
@container (max-width: 520px) {{
  .gf-stat {{ padding-left:1rem; padding-right:.5rem; }}
  .gf-v {{ font-size:1.6rem; }}
  .gf-lead .gf-v {{ font-size:2.9rem; }}
}}

/* ---- the days ahead as a calendar: one row per week */
.gf-calwrap {{ container-type:inline-size; }}
.gf-cal {{ display:grid; grid-template-columns:repeat(7, minmax(0,1fr)); gap:.5rem; }}
.gf-day {{
  position:relative; background:var(--gf-card); border-radius:.65rem; min-height:6.5rem;
  padding:.75rem 1.6rem .75rem .85rem; box-shadow:inset 0 0 0 1px var(--gf-line);
}}
.gf-day.gf-busier {{ background:var(--gf-hot-bg); box-shadow:none; }}
.gf-day.gf-quieter {{ background:var(--gf-cool-bg); box-shadow:none; }}
.gf-day.gf-closed {{ background:transparent; }}
.gf-when {{ font-size:.875rem; line-height:1.3; color:var(--gf-ink2); white-space:nowrap; }}
.gf-when b {{ color:var(--gf-ink); font-weight:600; }}
.gf-numrow {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:0 .45rem; margin-top:.3rem; }}
.gf-num {{ font-family:var(--gf-display); font-weight:800; font-variation-settings:"wdth" 78; font-size:2.15rem; line-height:1.1; color:var(--gf-ink); }}
.gf-chip {{ font-weight:700; font-size:.85rem; color:var(--gf-ink); white-space:nowrap; }}
.gf-closed .gf-num {{ font-size:1.2rem; font-weight:600; color:var(--gf-muted); margin-top:.6rem; }}
.gf-sub {{ font-size:.82rem; line-height:1.3; color:var(--gf-ink2); }}
.gf-gauge {{ position:absolute; right:.55rem; top:.75rem; bottom:.75rem; width:.45rem; border-radius:.25rem; background:rgba(16,32,29,.08); }}
.gf-gauge i {{ position:absolute; left:0; right:0; bottom:0; border-radius:.25rem; background:var(--gf-calm); }}
.gf-busier .gf-gauge i {{ background:var(--gf-hot); }}
.gf-quieter .gf-gauge i {{ background:var(--gf-cool); }}
.gf-gauge u {{ position:absolute; left:-.2rem; right:-.2rem; height:2px; margin-bottom:-1px; background:var(--gf-ink); border-radius:1px; }}
/* Below this width seven tiles a row get too narrow for a number and its flag side by side. */
@container (max-width: 1030px) {{
  .gf-cal {{ grid-template-columns:repeat(auto-fill, minmax(8.75rem, 1fr)); }}
}}

/* ---- chart keys */
.gf-key {{ display:flex; flex-wrap:wrap; gap:.3rem 1.4rem; font-size:.875rem; line-height:1.4; color:var(--gf-ink2); margin:.2rem 0 0; }}
.gf-key span {{ display:inline-flex; align-items:center; gap:.45rem; white-space:nowrap; }}
.gf-g {{ flex:none; position:relative; width:1.35rem; height:.9rem; }}
.gf-g::before {{ content:""; position:absolute; background:var(--c); }}
.gf-g-bar::before {{ left:.35rem; right:.35rem; top:0; bottom:0; border-radius:.16rem .16rem 0 0; }}
.gf-g-whisker::before {{ left:calc(50% - 1px); width:2px; top:0; bottom:0; }}
.gf-g-tick::before {{ left:.1rem; right:.1rem; top:calc(50% - 1px); height:2px; }}
.gf-g-ring::before {{ left:calc(50% - .3rem); top:calc(50% - .3rem); width:.6rem; height:.6rem; box-sizing:border-box; border-radius:50%; background:#fff; border:2px solid var(--c); }}
.gf-g-diamond::before {{ left:calc(50% - .28rem); top:calc(50% - .28rem); width:.56rem; height:.56rem; transform:rotate(45deg); }}
.gf-g-line::before {{ left:0; right:0; top:calc(50% - 1.5px); height:3px; border-radius:2px; }}
.gf-g-thin::before {{ left:0; right:0; top:calc(50% - 1px); height:2px; border-radius:1px; }}
.gf-g-dash::before {{ left:0; right:0; top:calc(50% - 1px); height:0; background:none; border-top:2px dashed var(--c); }}

/* ---- error compared, on the accuracy tab */
.gf-methods {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.8rem; margin:.6rem 0 1.2rem; }}
.gf-method {{ background:var(--gf-card); border:1px solid var(--gf-line); border-radius:10px; padding:.8rem 1rem; }}
.gf-method h4 {{ display:flex; align-items:center; gap:.5rem; margin:0 0 .35rem; padding:0; font-size:1rem; font-weight:650; color:var(--gf-ink); }}
.gf-method h4 i {{ width:.8rem; height:.8rem; border-radius:3px; background:var(--c); flex:none; }}
.gf-method p {{ margin:0; font-size:.92rem; line-height:1.45; color:var(--gf-ink2); }}
@media (max-width: 760px) {{ .gf-methods {{ grid-template-columns:minmax(0,1fr); }} }}
.gf-errs {{ display:grid; grid-template-columns:max-content minmax(4rem,1fr) max-content max-content; gap:.7rem 1rem; align-items:center; margin:.4rem 0 .2rem; }}
.gf-err-name {{ color:var(--gf-ink); font-weight:500; }}
.gf-err-track {{ height:1.5rem; }}
.gf-err-track b {{ display:block; height:100%; min-width:3px; border-radius:0 .25rem .25rem 0; }}
.gf-err-val {{ font-family:var(--gf-display); font-weight:700; font-variation-settings:"wdth" 82; font-size:1.5rem; line-height:1; color:var(--gf-ink); text-align:right; }}
.gf-err-val small {{ font-family:{BODY_FONT}; font-size:.8rem; font-weight:400; font-variation-settings:normal; color:var(--gf-muted); margin-left:.25rem; }}
.gf-err-note {{ color:var(--gf-ink2); font-size:.9rem; min-width:9.5rem; }}
@media (max-width: 700px) {{
  .gf-errs {{ grid-template-columns:minmax(0,1fr) max-content; }}
  .gf-err-name {{ grid-column:1 / -1; margin-bottom:-.5rem; }}
  .gf-err-note {{ grid-column:1 / -1; margin-top:-.5rem; min-width:0; }}
}}

.gf-foot {{ border-top:1px solid var(--gf-line); margin-top:1.5rem; padding-top:1.1rem; color:var(--gf-ink2); font-size:.9rem; line-height:1.5; }}
.gf-foot a {{ color:#12564A; text-underline-offset:3px; }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


# ---------------------------------------------------------------- header --------------------

def _story_plot() -> str:
    """The booking curve behind the method, drawn from STORY_THIS and STORY_NORMAL.

    st.html removes inline SVG, so the two lines travel as an image and the labels are
    ordinary text placed over it (in percentages, so they follow the image at any width).
    """
    w, h, left, right, top, bottom = 520, 196, 4, 128, 12, 32
    lo, hi = 100, 260
    n = len(STORY_THIS) - 1

    def x(i: float) -> float:
        return left + (w - left - right) * i / n

    def y(v: float) -> float:
        return top + (h - top - bottom) * (1 - (v - lo) / (hi - lo))

    def path(values: list[float]) -> str:
        return "M" + " L".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))

    points = [(x(i), y(v)) for i, v in enumerate(STORY_THIS)]
    length = sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                 for (x1, y1), (x2, y2) in zip(points, points[1:]))
    made = n - 6  # the forecast is made with 6 days to go
    xm, y_this, y_norm = x(made), y(STORY["booked"]), y(STORY["normally"])
    end = w - right
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">
<style>
.this {{ stroke-dasharray:{length:.0f}; stroke-dashoffset:{length:.0f}; animation:draw 1.5s .25s cubic-bezier(.4,0,.2,1) forwards; }}
.late {{ opacity:0; animation:show .5s 1.6s ease-out forwards; }}
@keyframes draw {{ to {{ stroke-dashoffset:0; }} }}
@keyframes show {{ to {{ opacity:1; }} }}
@media (prefers-reduced-motion: reduce) {{ .this {{ animation:none; stroke-dashoffset:0; }} .late {{ animation:none; opacity:1; }} }}
</style>
<line x1="{left}" x2="{end}" y1="{h - bottom}" y2="{h - bottom}" stroke="#3C6A5F"/>
<path d="{path(STORY_NORMAL)}" fill="none" stroke="#7FA79A" stroke-width="2" stroke-linejoin="round"/>
<path class="this" d="{path(STORY_THIS)}" fill="none" stroke="#FF7B52" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<g class="late">
<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{y_this + 8:.1f}" y2="{y_norm - 8:.1f}" stroke="{PAPER}" stroke-dasharray="3 3"/>
<circle cx="{xm:.1f}" cy="{y_this:.1f}" r="4.5" fill="#FF7B52" stroke="{BAND}" stroke-width="2"/>
<circle cx="{xm:.1f}" cy="{y_norm:.1f}" r="4.5" fill="#7FA79A" stroke="{BAND}" stroke-width="2"/>
</g>
</svg>"""
    src = "data:image/svg+xml;base64," + b64encode(svg.encode()).decode()

    def at(px: float, py: float) -> str:
        return f"left:{px / w * 100:.2f}%;top:{py / h * 100:.2f}%"

    gap = STORY["booked"] - STORY["normally"]
    alt = tr(f"Line chart: guests in the reservation book for Friday 28 April 2017 run well above "
             f"a normal Friday for the whole 60 days before arrival. A week ahead the book held "
             f"{STORY['booked']} guests where a normal Friday holds {STORY['normally']}.",
             f"Lijngrafiek: de gasten in het reserveringsboek voor vrijdag 28 april 2017 liggen de "
             f"hele 60 dagen voor aankomst ruim boven een normale vrijdag. Een week vooraf stonden er "
             f"{STORY['booked']} gasten in het boek, waar een normale vrijdag er {STORY['normally']} heeft.")
    return f"""<div class="gf-plot"><img src="{src}" alt="{alt}">
<span class="gf-tick" aria-hidden="true" style="{at(left, h - bottom + 9)}">{tr("60 days before", "60 dagen vooraf")}</span>
<span class="gf-tick" aria-hidden="true" style="{at(end, h - bottom + 9)};transform:translateX(-100%)">{tr("arrival", "aankomst")}</span>
<span class="gf-gap gf-late" aria-hidden="true" style="{at(xm - 9, (y_this + y_norm) / 2)}">+{gap} {tr("guests", "gasten")}</span>
<span class="gf-lab gf-late" aria-hidden="true" style="{at(end + 9, y(STORY_THIS[-1]))};color:#FFB59C">{tr("this Friday", "deze vrijdag")}</span>
<span class="gf-lab gf-late" aria-hidden="true" style="{at(end + 9, y(STORY_NORMAL[-1]))};color:#A9C7BC">{tr("a normal one", "een normale")}</span>
</div>"""


def hero(repo: str) -> str:
    s = STORY
    title = tr("How many guests are coming?", "Hoeveel gasten komen er?")
    lede = tr("A forecast for the next two weeks that reads your reservation book: the usual number "
              "for each weekday, corrected by how far bookings are running ahead of or behind normal.",
              "Een voorspelling voor de komende twee weken die je reserveringsboek leest: het "
              "gebruikelijke aantal per weekdag, bijgesteld naar hoeveel de boekingen voor of achter "
              "lopen op normaal.")
    code = tr("Code on GitHub", "Code op GitHub")
    caption = tr(
        f"A week ahead of <b>Friday 28 April 2017</b>, one hotel's book held <b>{s['booked']} guests</b>; "
        f"a normal Friday holds {s['normally']}. The weekday average said {s['weekday_average']}. "
        f"This forecast said <b>{s['pace']}</b>, and <b>{s['actual']} came</b>.",
        f"Een week voor <b>vrijdag 28 april 2017</b> stonden er in het boek van één hotel "
        f"<b>{s['booked']} gasten</b>; op een normale vrijdag zijn dat er {s['normally']}. Het "
        f"weekdaggemiddelde zei {s['weekday_average']}. Deze voorspelling zei <b>{s['pace']}</b>, "
        f"en er kwamen er <b>{s['actual']}</b>.")
    return f"""
<header class="gf-hero">
  <div>
    <h1>{title}</h1>
    <p class="gf-lede">{lede}</p>
    <p class="gf-cta"><a class="gf-code" href="{escape(repo)}" target="_blank" rel="noopener">{code}</a></p>
  </div>
  <figure class="gf-story">{_story_plot()}
    <p class="gf-caption">{caption}</p>
  </figure>
</header>"""


# ---------------------------------------------------------------- building blocks -----------

def stats(items: list[tuple[str, str, str]]) -> str:
    """A row of headline figures: (value, what it is, a comparison). The first one leads."""
    cells = []
    for i, (value, label, detail) in enumerate(items):
        lead = " gf-lead" if i == 0 else ""
        cells.append(f'<div class="gf-stat{lead}"><div class="gf-v">{escape(value)}</div>'
                     f'<div class="gf-l">{escape(label)}</div><div class="gf-d">{escape(detail)}</div></div>')
    return f'<div class="gf-stats">{"".join(cells)}</div>'


def day_status(vs_usual_pct: float, unusual: float) -> str:
    """'busier', 'quieter' or 'usual': the one rule behind every colour in the app."""
    if vs_usual_pct >= unusual:
        return "busier"
    if vs_usual_pct <= -unusual:
        return "quieter"
    return "usual"


def calendar(days: list[dict]) -> str:
    """The days ahead as tiles, seven to a row so each column is one weekday.

    Each day is a dict with: date (Timestamp), closed, forecast, usual, status, vs_usual_pct,
    sub (a short line under the number) and title (the hover text).
    The bar at the right of a tile is the forecast; the line across it is the usual number.
    """
    open_days = [d for d in days if not d["closed"]]
    top = max([max(d["forecast"], d["usual"]) for d in open_days] or [1]) or 1
    tiles = []
    for d in days:
        day, date = short_date(d["date"]).split(" ", 1)
        when = f'<b>{day}</b> {date}'
        if d["closed"]:
            tiles.append(f'<div class="gf-day gf-closed"><div class="gf-when">{when}</div>'
                         f'<div class="gf-numrow"><span class="gf-num">{tr("Closed", "Gesloten")}</span></div></div>')
            continue
        chip = ""
        if d["status"] != "usual":
            arrow = "↑" if d["status"] == "busier" else "↓"
            chip = f'<span class="gf-chip">{arrow} {abs(d["vs_usual_pct"]):.0f}%</span>'
        tiles.append(
            f'<div class="gf-day gf-{d["status"]}" title="{escape(d["title"])}">'
            f'<div class="gf-when">{when}</div>'
            f'<div class="gf-numrow"><span class="gf-num">{d["forecast"]:.0f}</span>{chip}</div>'
            f'<div class="gf-sub">{escape(d["sub"])}</div>'
            f'<div class="gf-gauge"><i style="height:{min(100, d["forecast"] / top * 100):.0f}%"></i>'
            f'<u style="bottom:{min(100, d["usual"] / top * 100):.0f}%"></u></div></div>')
    return f'<div class="gf-calwrap"><div class="gf-cal">{"".join(tiles)}</div></div>'


GLYPHS = ("bar", "whisker", "tick", "ring", "diamond", "line", "thin", "dash")


def key(items: list[tuple[str, str, str]]) -> str:
    """A chart key: (glyph, colour, label). The glyph is drawn the way the chart draws it."""
    parts = []
    for glyph, colour, label in items:
        assert glyph in GLYPHS, glyph
        parts.append(f'<span><i class="gf-g gf-g-{glyph}" style="--c:{colour}" aria-hidden="true"></i>'
                     f'{escape(label)}</span>')
    return f'<div class="gf-key">{"".join(parts)}</div>'


def error_bars(rows: list[tuple[str, float, str, str]]) -> str:
    """Average error side by side: (name, guests off per day, colour, note). Shorter is better."""
    longest = max(value for _, value, _, _ in rows) or 1
    cells = []
    for name, value, colour, note in rows:
        cells.append(
            f'<div class="gf-err-name">{escape(name)}</div>'
            f'<div class="gf-err-track"><b style="width:{value / longest * 100:.1f}%;background:{colour}"></b></div>'
            f'<div class="gf-err-val">{tr(f"{value:.1f}", f"{value:.1f}".replace(".", ","))}'
            f'<small>{tr("guests", "gasten")}</small></div>'
            f'<div class="gf-err-note">{escape(note)}</div>')
    return f'<div class="gf-errs">{"".join(cells)}</div>'


def methods(items: list[tuple[str, str, str]]) -> str:
    """The forecasting methods side by side: (colour, name, one plain sentence)."""
    cards = [f'<div class="gf-method"><h4><i style="--c:{colour}" aria-hidden="true"></i>{escape(name)}</h4>'
             f'<p>{escape(text)}</p></div>' for colour, name, text in items]
    return f'<div class="gf-methods">{"".join(cards)}</div>'


def footer(repo: str) -> str:
    link = f'<a href="{escape(repo)}" target="_blank" rel="noopener">'
    return tr(f'<footer class="gf-foot">Built by Rebeka Olar. {link}Code and tests on GitHub</a>. '
              'Your file is processed in memory and not saved.</footer>',
              f'<footer class="gf-foot">Gemaakt door Rebeka Olar. {link}Code en tests op GitHub</a>. '
              'Je bestand wordt alleen in het geheugen verwerkt en niet opgeslagen.</footer>')


# ---------------------------------------------------------------- charts --------------------

def polish(chart, height: int):
    """The shared chart style: quiet axes and grid, the page's own font, no frame."""
    axis = dict(labelFont=BODY_FONT, titleFont=BODY_FONT, labelColor=INK2, titleColor=INK2,
                labelFontSize=12, titleFontSize=12, titleFontWeight=500, titlePadding=10,
                gridColor=LINE, gridWidth=1, domainColor="#BCC7C2", tickColor="#BCC7C2", labelPadding=6)
    locale = vega_locale()  # Dutch day and month names on the axes and in tooltips
    return (chart.properties(height=height, background="transparent")
            .configure(font=BODY_FONT, **({"locale": locale} if locale else {}))
            .configure_view(stroke=None)
            .configure_axis(**axis)
            .configure_axisY(domain=False, ticks=False, labelColor=MUTED)
            .configure_axisX(grid=False, tickSize=4))
