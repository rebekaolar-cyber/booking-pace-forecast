"""Scoring functions for the assistant eval. Pure functions, no API calls (except ``judge``).

The core metric is *number faithfulness*: every number the assistant writes must come from a
tool output of the same run, for the day the answer is talking about.

How it works (deterministic, no model involved):

1. The tool outputs are split into *records* (every dict with a ``"date"`` key: one forecast
   day, one staffing result) and *undated* values (accuracy, today info, threshold, numbers
   inside text such as "1 per 40 guests", list lengths such as "3 flagged days").
2. The answer is split into clauses (sentences, bullet/table lines, semicolons; a comma or
   "and" also splits when both sides name their own day). For each clause we find the days it
   names: "2017-06-03", "3 June", "Sat 3 Jun", "zaterdag 3 juni", "today", "morgen", or a
   weekday name resolved against the days the tools returned. A clause that names no day
   inherits the previous clause's days.
3. A number in a clause is grounded only if it matches a value from the records of *those*
   days, an undated value, or a number from the question. Signs count ("4% busier" must match
   +4), ranges must match a low/high pair, rounding is allowed (8.6 -> 9, 8.5 -> 9).

If the whole answer names no day and the run touched several days, every record is used and
the result is marked ``loose`` so it is visible in the eval output.

Special cases, each with its own small rule:

* The flag threshold (15%) is allowed only where the text says it is the threshold ("at least
  15%", "15% or more", "the 15% flag", "drempel"), never as a day's own change. Rounding may
  not move a day across it: -14.5% may be written "14.5%" but not "15% quieter".
* The range percentile (80, "8 in 10 past days stayed within") is how the range is built, not
  how often it held. It grounds a number only in a clause that defines the range.
* "N days" in a clause about flags/busy/quiet days must be a count the tools returned (all
  flagged, busy or quiet days) or the number of days the answer itself lists.
* Totals and averages are computed only over days the sentence names. In a loose "on average"
  clause only usual-for-weekday values and undated accuracy values count.

Every check returns a plain bool or a small dict, so a failing question can be explained by
looking at its JSON record.
"""
from __future__ import annotations

KNOWN_LIMITS = """\
Known limits of the number-faithfulness check (it is a deterministic text check, not a judge):
- A loose clause (no day named, several days fetched) is checked against every day of the run,
  so a number from the wrong day can pass there. Such answers are marked "loose".
- A count of days is only checked when it is written as "N (busy/flagged/...) days" in a
  clause about flags. "Of the 28 days, 3 are flagged" is not checked as a count.
- Averages and totals are accepted when they are arithmetically right for the days the
  sentence names; an average over unnamed days ("the next two weeks") is not computed and fails.
- Numbers repeated from the question always pass, even when the answer uses them for another
  meaning.
- The range percentile (80) is told apart from a coverage claim by wording ("stayed within",
  "percentile" vs "held", "landed", "inside"). Unusual wording can be misread either way.
- Language detection counts common words; a very short answer with no day or month name can
  still come out "unknown".
"""

import datetime as dt
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from agent.contract import FLAG_THRESHOLD_PCT, MAX_HORIZON_DAYS, RunResult

# =========================================================================== vocabulary

_MONTH_NAMES = {
    "january": 1, "jan": 1, "januari": 1, "february": 2, "feb": 2, "februari": 2,
    "march": 3, "mar": 3, "maart": 3, "mrt": 3, "april": 4, "apr": 4, "may": 5, "mei": 5,
    "june": 6, "jun": 6, "juni": 6, "july": 7, "jul": 7, "juli": 7, "august": 8, "aug": 8,
    "augustus": 8, "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10,
    "oktober": 10, "okt": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTHS = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
_DAY = r"(?:3[01]|[12]\d|0?[1-9])"  # 1..31
_ORD = r"(?:st|nd|rd|th|ste|de|e)?"  # 3rd, 3e, 3de, 1ste
_RANGE_JOIN = r"(?:tot en met|t/m|through|until|till|to|tot|-|–|—)"
_LIST_JOIN = r"(?:and|en|&|,)"

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5,
    "sunday": 6, "maandag": 0, "dinsdag": 1, "woensdag": 2, "donderdag": 3, "vrijdag": 4,
    "zaterdag": 5, "zondag": 6,
}
# Short forms only when capitalised ("Sat 3 Jun"), so the verb "sat" or "sun" is not a day.
# Dutch two-letter forms (za, zo, do, ...) are ordinary words and are not used.
_WEEKDAY_ABBR = {"Mon": 0, "Tue": 1, "Tues": 1, "Wed": 2, "Thu": 3, "Thur": 3, "Thurs": 3,
                 "Fri": 4, "Sat": 5, "Sun": 6}
_NEXT_WORDS = {"next", "coming", "volgende", "komende", "aanstaande"}
_THIS_WORDS = {"this", "deze", "dit"}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    # Dutch. "een" is left out on purpose: it is also the article "a".
    "nul": 0, "één": 1, "eén": 1, "twee": 2, "drie": 3, "vier": 4, "vijf": 5, "zes": 6,
    "zeven": 7, "acht": 8, "negen": 9, "tien": 10, "elf": 11, "twaalf": 12, "dertien": 13,
    "veertien": 14, "vijftien": 15, "zestien": 16, "zeventien": 17, "achttien": 18,
    "negentien": 19, "twintig": 20,
}
_NOT_A_COUNT_BEFORE_ONE = {"the", "this", "that", "which", "no", "any", "every", "each", "a",
                           "an", "someone", "anyone", "everyone"}

# Words that follow a quantity. "3-6 guests" or "June 3 - 12 extra" are numbers, not dates.
_QUANTITY_WORDS = {
    "guest", "guests", "gast", "gasten", "people", "personen", "mensen", "covers", "staff",
    "personeel", "bartender", "bartenders", "cook", "cooks", "kok", "koks", "waiter",
    "waiters", "server", "servers", "ober", "obers", "medewerker", "medewerkers", "extra",
    "more", "fewer", "less", "meer", "minder", "percent", "procent", "bar", "kitchen",
    "floor", "keuken", "bediening", "barpersoneel", "keukenpersoneel", "barmedewerkers",
    "bookings", "boekingen", "reservations", "reserveringen", "tables", "tafels", "shifts",
}
_UP_WORDS = {"busier", "more", "extra", "higher", "above", "up", "increase", "drukker", "meer",
             "hoger", "boven", "plus", "stijging"}
_DOWN_WORDS = {"quieter", "fewer", "less", "lower", "below", "under", "down", "decrease",
               "drop", "rustiger", "minder", "lager", "onder", "daling"}
_TOTAL_RE = re.compile(r"\b(?:total|totaal|in all|altogether|combined|samen|opgeteld|sum)\b", re.I)
_AVERAGE_RE = re.compile(r"\b(?:average|avg|mean|gemiddeld\w*)\b", re.I)

# Role words, so "4 bartenders" is not grounded by the kitchen's 4. Stems match word starts.
_ROLE_STEMS = {
    "bar": ("bar",),
    "kitchen": ("kitchen", "keuken", "cook", "kok", "chef"),
    "floor": ("floor", "waiter", "waitress", "server", "ober", "bediening", "zaal"),
}

# Public facts from the tool descriptions the model always sees, allowed only in context.
# (The 15% flag threshold has its own, stricter rule: see _threshold_ok.)
_ALWAYS = [(float(MAX_HORIZON_DAYS), ("days", "day", "dagen", "dag")),
           (float(MAX_HORIZON_DAYS // 7), ("weeks", "weken"))]

# "15%" may be the flag threshold only where the text says so. Either the words right next to
# the number say it ("at least 15%", "15% or more", "threshold of 15%"), or the number has no
# busier/quieter direction and the clause is about the threshold ("just under the 15% flag").
_THRESHOLD_BEFORE = re.compile(r"(?:at least|minstens|minimaal|ten minste|threshold|drempel|cut-?off|grens)\b[\w\s]{0,12}$", re.I)
_THRESHOLD_AFTER = re.compile(r"^\s*(?:%|procent|percent)?\s*(?:or more|of meer|and up|en hoger)\b",
                              re.I)
_THRESHOLD_TOPIC = re.compile(r"threshold|drempel|\bflag|gemarkeerd|markering|\bgrens", re.I)

# The likely range's percentile (80: "8 in 10 past days stayed within") says how the range is
# BUILT. It is not how often the range held the real number afterwards, so it may only ground a
# number in a clause that defines the range, never in a coverage claim ("held 80% of days").
_DEFINITION_KEY = "range_percentile"
_DEFINITION_TEXT = re.compile(r"percentil|stayed within|stay within|stays within|"
                              r"bleven binnen|blijven binnen|binnen bleven", re.I)
_COVERAGE_TEXT = re.compile(r"\bheld\b|\bholds?\b|\bland(?:ed|s)?\b|\blag(?:en)? binnen|"
                            r"real number|actual|echte|werkelijk|klopte|was right|"
                            r"\binside\b|\bfell\b|\bfalls?\b|covered|coverage", re.I)

# Counts of days in a clause about flags ("3 busy days", "12 days are flagged").
_DAYS_AFTER = re.compile(r"^\s*(?:[^\W\d_]+\s+){0,2}?(?:days?|dagen|dag)\b", re.I)
_WINDOW_BEFORE = {"next", "coming", "komende", "volgende", "first", "eerste", "last", "laatste",
                  "past", "afgelopen", "over", "within", "binnen", "in", "of", "van", "all", "alle"}
_FLAG_TOPIC = {
    "flagged": re.compile(r"\bflag|gemarkeerd|unusual|opvallend|afwijkend", re.I),
    "busy": re.compile(r"\bbus(?:y|ier)\b|\bdruk(?:ke|ker)?\b", re.I),
    "quiet": re.compile(r"\bquiet(?:er)?\b|\brustig(?:e|er)?\b", re.I),
}

# Pairs of fields whose difference an answer may state ("5 fewer than usual", "11 still to come",
# "give or take 10").
_DERIVED = [("forecast_guests", "usual_for_weekday_guests"),
            ("forecast_guests", "booked_so_far_guests"),
            ("range_high_guests", "forecast_guests"),
            ("forecast_guests", "range_low_guests")]
_FORECAST_TOOLS = {"get_forecast", "get_flagged_days", "get_staffing"}

# =========================================================================== small helpers


def _to_float(token: str) -> float:
    """'1,5' and '1.5' -> 1.5; '1,200' and '1.200' -> 1200 (a 3-digit group is a thousands group)."""
    parts = re.split(r"[.,]", token)
    if len(parts) == 1:
        return float(token)
    if all(len(p) == 3 for p in parts[1:]):
        return float("".join(parts))
    return float(parts[0] + "." + "".join(parts[1:]))


def _round_half_up(x: float, places: int) -> float:
    q = Decimal(1).scaleb(-places)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))  # -8.5 -> -9


def value_matches(answer: float, source: float) -> bool:
    """Equal, or the answer is the source rounded to a whole number or one decimal
    (half-up: 8.6 -> 9, 8.5 -> 9, 8.4 -> 8)."""
    return any(abs(answer - s) < 1e-6
               for s in (source, _round_half_up(source, 0), _round_half_up(source, 1)))


def _iso(value: Any) -> dt.date | None:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            return dt.date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _mkdate(y: int, m: int, d: int) -> dt.date | None:
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _word_after(text: str, pos: int) -> str:
    m = re.match(r"[\s\-–(]*([^\W\d_]+|%)", text[pos:])
    return m.group(1).lower() if m else ""


def _words_before(text: str, pos: int, n: int) -> list[str]:
    return [w.lower() for w in re.findall(r"[^\W\d_]+", text[max(0, pos - 40):pos])][-n:]


# =========================================================================== tool outputs

@dataclass
class _Value:
    value: float
    key: str
    role: str | None = None
    dated: bool = False


@dataclass
class _Sources:
    records: dict[dt.date, list[_Value]] = field(default_factory=dict)
    pairs: dict[dt.date, list[tuple[float, float]]] = field(default_factory=dict)
    undated: list[_Value] = field(default_factory=list)
    definition: list[float] = field(default_factory=list)  # range percentile: see _DEFINITION_KEY
    flag_counts: dict[str, set[float]] = field(default_factory=dict)  # "busy" -> {3, 8}
    forecast_values: list[float] = field(default_factory=list)  # leaves of forecast/staffing records
    today: dt.date | None = None
    years: set[int] = field(default_factory=set)


@dataclass
class _Ctx:
    run_dates: set[dt.date]
    today: dt.date | None
    years: set[int]


def _collect(result: RunResult) -> _Sources:
    src = _Sources()
    flags_by_day: dict[dt.date, str | None] = {}
    for call in result.tool_calls:
        out = call.output
        if isinstance(out, dict) and _iso(out.get("today")):
            src.today = _iso(out["today"])
        _walk(out, "", None, None, call.name, src)
        days = out.get("days") if isinstance(out, dict) else None
        if isinstance(days, list):  # how many days each call flagged: "3 busy days"
            flags = {_iso(d.get("date")): d.get("flag") for d in days if isinstance(d, dict)}
            flags_by_day.update(flags)
            _add_flag_counts(src, list(flags.values()))
    if flags_by_day:  # and over all calls together
        _add_flag_counts(src, list(flags_by_day.values()))
    src.years |= {d.year for d in src.records}
    if src.today:
        src.years.add(src.today.year)
    return src


def _add_flag_counts(src: _Sources, flags: list[str | None]) -> None:
    counts = {"flagged": sum(f is not None for f in flags),
              "busy": sum(f == "busy" for f in flags), "quiet": sum(f == "quiet" for f in flags)}
    for topic, n in counts.items():
        src.flag_counts.setdefault(topic, set()).add(float(n))


def _walk(value: Any, key: str, day: dt.date | None, role: str | None, tool: str,
          src: _Sources) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if key == _DEFINITION_KEY:
            src.definition.append(float(value))
        elif "threshold" in key:
            pass  # the 15% threshold is only allowed in context, see _threshold_ok
        elif day is None:
            src.undated.append(_Value(float(value), key))
        else:
            src.records[day].append(_Value(float(value), key, role, dated=True))
            if tool in _FORECAST_TOOLS:
                src.forecast_values.append(float(value))
        return
    if isinstance(value, str):
        d = _iso(value)
        if d:
            src.years.add(d.year)
        elif _DEFINITION_TEXT.search(value):  # "8 in 10 past days stayed within ..."
            src.definition.extend(numbers_in(value))
        else:  # rule text, error messages: their numbers are undated facts
            src.undated.extend(_Value(n, key) for n in numbers_in(value))
        return
    if isinstance(value, (list, tuple)):
        src.undated.append(_Value(float(len(value)), "number_of_" + key))  # "2 flagged days"
        for v in value:
            _walk(v, key, day, role, tool, src)
        return
    if isinstance(value, dict):
        if day is None and _iso(value.get("date")):
            day = _iso(value["date"])
            src.years.add(day.year)
            src.records.setdefault(day, [])
            nums = {k: float(v) for k, v in value.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for k, v in nums.items():  # low/high pairs, e.g. range_low_guests / range_high_guests
                hi = k.replace("low", "high")
                if "low" in k and hi in nums:
                    src.pairs.setdefault(day, []).append((v, nums[hi]))
            for a, b in _DERIVED:
                if a in nums and b in nums:
                    src.records[day].append(_Value(nums[a] - nums[b], "derived"))
        if isinstance(value.get("role"), str):
            role = value["role"].strip().lower()
        for k, v in value.items():
            if k != "date":
                _walk(v, k, day, role, tool, src)


def _ctx_for(src: _Sources) -> _Ctx:
    return _Ctx(set(src.records), src.today, set(src.years))


def _empty_ctx() -> _Ctx:
    return _Ctx(set(), None, set())


# =========================================================================== text scanning

@dataclass
class _DateMention:
    start: int
    end: int
    dates: list[dt.date]
    weak: bool = False  # a weekday name: only used when the clause names no explicit date


@dataclass
class _Num:
    start: int          # incl. sign
    digits: int         # where the digits (or the number word) start
    end: int
    value: float        # always >= 0; the sign is kept separately
    sign: int = 0       # +1 / -1 when the text says so ("+4%", "4% busier", "5 fewer")
    pct: bool = False
    after: str = ""     # first word after the number (or "%")
    always: bool = False  # 28 days / 4 weeks
    threshold: bool = False  # "15%": may be the flag threshold, see _threshold_ok


@dataclass
class _Item:
    kind: str           # "single" | "range" | "fraction"
    nums: list[_Num]

    @property
    def start(self) -> int:
        return self.nums[0].start


def _resolve_dm(ctx: _Ctx, d: int, m: int, y: int | None = None) -> dt.date | None:
    if y is not None:
        return _mkdate(y if y > 99 else 2000 + y, m, d)
    for yy in sorted(ctx.years):
        cand = _mkdate(yy, m, d)
        if cand in ctx.run_dates:
            return cand
    yy = ctx.today.year if ctx.today else (min(ctx.years) if ctx.years else 2000)
    return _mkdate(yy, m, d)


def _expand(a: dt.date, b: dt.date) -> list[dt.date]:
    if not a or not b or b < a or (b - a).days > 62:
        return [x for x in (a, b) if x]
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def _day_list(spec: str, between: bool) -> list[int | tuple[int, int]]:
    """'17 and 18' -> [17, 18]; '5 to 11' -> [(5, 11)]; 'between 1 and 14' -> [(1, 14)]."""
    tokens = re.findall(rf"(\d{{1,2}}){_ORD}|({_RANGE_JOIN}|{_LIST_JOIN})", spec, re.I)
    out: list[int | tuple[int, int]] = []
    pending_range = False
    for num, join in tokens:
        if join:
            pending_range = bool(re.fullmatch(_RANGE_JOIN, join, re.I)) or (
                between and join.lower() in ("and", "en", "&"))
            continue
        n = int(num)
        if pending_range and out and isinstance(out[-1], int):
            out[-1] = (out[-1], n)
        else:
            out.append(n)
        pending_range = False
    return out


class _Scanner:
    """Finds date mentions and numbers in one text. Date-like parts (dates, clock times, week
    numbers, list markers, ordinals, years of known dates) are never counted as numbers."""

    def __init__(self, text: str, ctx: _Ctx):
        self.text = text
        self.ctx = ctx
        self.taken: list[tuple[int, int]] = []
        self.dates: list[_DateMention] = []
        self._find_dates()
        self.nums = self._find_numbers()
        self.items = self._pair(self.nums)

    # ------------------------------------------------------------------ helpers
    def _free(self, s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in self.taken)

    def _take(self, s: int, e: int, dates: list[dt.date] | None = None, weak: bool = False):
        self.taken.append((s, e))
        if dates is not None:
            self.dates.append(_DateMention(s, e, [d for d in dates if d], weak))

    def _scan(self, pattern: str, flags=re.I):
        for m in re.finditer(pattern, self.text, flags):
            if self._free(m.start(), m.end()):
                yield m

    def _quantity_after(self, pos: int) -> bool:
        return _word_after(self.text, pos) in _QUANTITY_WORDS | {"%"}

    # ------------------------------------------------------------------ dates
    def _find_dates(self) -> None:
        t, ctx = self.text, self.ctx
        for m in self._scan(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)"):  # 2017-06-03
            self._take(m.start(), m.end(), [_mkdate(*map(int, m.groups()))])
        for m in self._scan(r"(?<![\d.,/-])(\d{1,2})([-./])(\d{1,2})\2((?:19|20)?\d{2})(?!\d)"):
            self._take(m.start(), m.end(),                                    # 03-06-2017, 3/6/17
                       [_resolve_dm(ctx, int(m[1]), int(m[3]), int(m[4]))])
        for m in self._scan(r"(?<![\d.,:])\d{1,2}:\d{2}(?!\d)"                 # 18:00
                            r"|(?<![\d.,])\d{1,2}\.\d{2}\s*(?:uur|u|h)\b"      # 18.00 uur
                            r"|(?<![\d.,])\d{1,2}\s*(?:am|pm|uur)\b"):         # 6 pm
            self._take(m.start(), m.end())
        # 3 June, 3rd of June, 17 and 18 June, 19 t/m 25 juni, 5 to 11 June 2017
        day_first = (rf"(?<![\d.,])(?P<days>{_DAY}{_ORD}(?:\s*(?:{_RANGE_JOIN}|{_LIST_JOIN})\s*"
                     rf"{_DAY}{_ORD})*)\s+(?:of\s+)?(?P<month>{_MONTHS})\b"
                     r"(?:,?\s+(?P<year>(?:19|20)\d{2})(?!\d))?")
        for m in self._scan(day_first):
            between = bool(re.search(r"\b(?:between|tussen)\s*$", t[:m.start()], re.I))
            month = _MONTH_NAMES[m["month"].lower()]
            year = int(m["year"]) if m["year"] else None
            out: list[dt.date] = []
            for d in _day_list(m["days"], between):
                if isinstance(d, tuple):
                    out += _expand(_resolve_dm(ctx, d[0], month, year), _resolve_dm(ctx, d[1], month, year))
                else:
                    out.append(_resolve_dm(ctx, d, month, year))
            self._take(m.start(), m.end(), out)
        # June 3, June 3rd, 2017, June 3-4 (but not "June 3 - 12 extra guests")
        for m in self._scan(rf"\b(?P<month>{_MONTHS})\.?\s+(?P<d1>{_DAY}){_ORD}(?![\d.,]?\d)"):
            month, d1, end = _MONTH_NAMES[m["month"].lower()], int(m["d1"]), m.end()
            days = [d1]
            c = re.match(rf"\s*({_RANGE_JOIN}|and|en|&)\s*({_DAY}){_ORD}(?!\d)", t[end:], re.I)
            if c and int(c[2]) > d1 and not self._quantity_after(end + c.end()):
                days = list(range(d1, int(c[2]) + 1)) if re.fullmatch(_RANGE_JOIN, c[1], re.I) \
                    else [d1, int(c[2])]
                end += c.end()
            y = re.match(r",?\s+((?:19|20)\d{2})(?!\d)", t[end:])
            year = int(y[1]) if y else None
            if y:
                end += y.end()
            if self._free(m.start(), end):
                self._take(m.start(), end, [_resolve_dm(ctx, d, month, year) for d in days])
        for m in self._scan(rf"\b(?:{_MONTHS})\s+(?:19|20)\d{{2}}(?!\d)"):  # August 2017
            self._take(m.start(), m.end())
        for m in self._scan(r"(?<![\d.,/])(\d{1,2})/(\d{1,2})(?![\d/]|[.,]\d)"):  # 03/06
            a, b = int(m[1]), int(m[2])
            d = _resolve_dm(ctx, a, b)
            alt = _resolve_dm(ctx, b, a)
            if d not in ctx.run_dates and alt in ctx.run_dates:
                d = alt
            if d or alt:
                self._take(m.start(), m.end(), [d or alt])
        for m in self._scan(r"(?<![\d.,/\-–])(\d{1,2})-(\d{1,2})(?![\d\-–]|[.,]\d)"):  # 3-6 (juni)
            d = _resolve_dm(ctx, int(m[1]), int(m[2]))
            if d in ctx.run_dates and not self._quantity_after(m.end()):
                self._take(m.start(), m.end(), [d])
        for m in self._scan(r"(?<![\d.,])(\d{1,2})(?:st|nd|rd|th|ste|de|e)\b"):  # the 10th
            prev = _words_before(t, m.start(), 1)
            nxt = _word_after(t, m.end())
            dates = None
            if prev and prev[0] in {"the", "on", "op", "de", "from", "to", "until", "tot"} \
                    and not re.search(r"(?:est|ste)$", nxt):
                dates = sorted(d for d in ctx.run_dates if d.day == int(m[1]))
            self._take(m.start(), m.end(), dates)
        for m in self._scan(r"\b(?:week|wk|weeknummer)\s*(?:nr\.?\s*|no\.?\s*|#)?\d{1,2}\b"):
            self._take(m.start(), m.end())  # week 23
        for m in self._scan(r"(?<!\d )\b(?:in|over)\s+(\d{1,2})\s+(?:days|dagen)\b"):  # in 10 days
            when = [ctx.today + dt.timedelta(days=int(m[1]))] if ctx.today else []
            self._take(m.start(), m.end(), when)
        for m in self._scan(r"(?m)^[ \t>*\-]*(\d{1,2}[.)])(?=\s)"):  # "1. " list markers
            self._take(m.start(1), m.end(1))
        for m in self._scan(r"(?<![\d.,])((?:19|20)\d{2})(?!\d|[.,]\d)"):  # 2017 (a known year)
            if int(m[1]) in ctx.years:
                self._take(m.start(), m.end())
        # Relative days and weekday names: words only, so they never hide a number.
        rel = {"day after tomorrow": 2, "overmorgen": 2, "today": 0, "tonight": 0, "vandaag": 0,
               "vanavond": 0, "tomorrow": 1, "morgen": 1}
        for m in re.finditer(r"\b(day after tomorrow|overmorgen|today|tonight|vandaag|vanavond|"
                             r"tomorrow|morgen)\b", t, re.I):
            if self._free(m.start(), m.end()):
                when = [ctx.today + dt.timedelta(days=rel[m[1].lower()])] if ctx.today else []
                self._take(m.start(), m.end(), when)
        names = "|".join(_WEEKDAYS)
        for m in re.finditer(rf"\b(?:(?P<mod>this coming|next|coming|this|volgende|komende|"
                             rf"aanstaande|deze|dit)\s+)?(?P<wd>{names}|weekend|weekeinde)(?:s|en)?\b",
                             t, re.I):
            wd = m["wd"].lower()
            wds = {5, 6} if wd.startswith("week") else {_WEEKDAYS[wd]}
            self.dates.append(_DateMention(m.start(), m.end(), self._weekday(wds, m["mod"]), True))
        abbr = "|".join(sorted(_WEEKDAY_ABBR, key=len, reverse=True))
        for m in re.finditer(rf"\b({abbr})\b\.?", t):
            self.dates.append(_DateMention(m.start(), m.end(),
                                           self._weekday({_WEEKDAY_ABBR[m[1]]}, None), True))
        # "from 5 June to 11 June", "between 2017-06-02 and 2017-06-28"
        strong = sorted((d for d in self.dates if not d.weak and len(d.dates) == 1),
                        key=lambda d: d.start)
        for a, b in zip(strong, strong[1:]):
            sep = t[a.end:b.start]
            between = re.search(r"\b(?:between|tussen)\s*$", t[:a.start], re.I)
            if re.fullmatch(rf"\s*{_RANGE_JOIN}\s*", sep, re.I) or (
                    between and re.fullmatch(r"\s*(?:and|en|&)\s*", sep, re.I)):
                a.dates = _expand(a.dates[0], b.dates[0])

    def _weekday(self, weekdays: set[int], mod: str | None) -> list[dt.date]:
        ctx = self.ctx
        cands = sorted(d for d in ctx.run_dates if d.weekday() in weekdays)
        mod = (mod or "").lower()
        if not mod or ctx.today is None:
            return cands
        after = ctx.today if mod.split()[0] in _THIS_WORDS and "coming" not in mod \
            else ctx.today + dt.timedelta(days=1)
        first = next((ctx.today + dt.timedelta(days=i) for i in range(14)
                      if ctx.today + dt.timedelta(days=i) >= after
                      and (ctx.today + dt.timedelta(days=i)).weekday() in weekdays), None)
        if first is None:
            return cands
        if weekdays == {5, 6}:  # the weekend: that Saturday and Sunday
            return [first + dt.timedelta(days=i) for i in range(2)
                    if (first + dt.timedelta(days=i)).weekday() in weekdays]
        return [first]

    # ------------------------------------------------------------------ numbers
    def _find_numbers(self) -> list[_Num]:
        t = self.text
        nums: list[_Num] = []
        for m in re.finditer(r"(?<![\w.,])(?P<sign>[+\-−]?)(?P<num>\d+(?:[.,]\d+)*)"
                             r"|(?<![^\W\d_])[x×](?P<xnum>\d+(?:[.,]\d+)*)", t):
            token = m["num"] or m["xnum"]
            digits = m.start("num") if m["num"] else m.start("xnum")
            if not self._free(digits, digits + len(token)):
                continue
            sign = {"+": 1, "-": -1, "−": -1}.get(m["sign"] or "", 0)
            nums.append(self._finish(m.start(), digits, digits + len(token), _to_float(token), sign))
        words = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
        for m in re.finditer(rf"\b({words})\b(?!-)", t, re.I):
            if not self._free(m.start(), m.end()):
                continue
            w = m[1].lower()
            if w == "one" and (set(_words_before(t, m.start(), 1)) & _NOT_A_COUNT_BEFORE_ONE
                               or _word_after(t, m.end()) in {"of", "another"}):
                continue
            nums.append(self._finish(m.start(), m.start(), m.end(), float(_NUMBER_WORDS[w]), 0))
        return sorted(nums, key=lambda n: n.start)

    def _finish(self, start: int, digits: int, end: int, value: float, sign: int) -> _Num:
        t = self.text
        p = re.match(r"\s*(%|per ?cent\b|procent\b|pct\b)", t[end:], re.I)
        after_pos = end + (p.end() if p else 0)
        n = _Num(start, digits, end, value, sign, pct=bool(p), after=_word_after(t, after_pos))
        if p:
            n.after = "%"
        if not sign:
            following = re.findall(r"[^\W\d_]+", t[after_pos:after_pos + 30].lower())
            look = following[:2] if p else following[:1]
            before = _words_before(t, start, 2) if p else []
            if any(w in _UP_WORDS for w in look) or "up" in before:
                n.sign = 1
            elif any(w in _DOWN_WORDS for w in look) or "down" in before:
                n.sign = -1
        n.always = any(abs(value - v) < 1e-9 and (n.after in ctx_words)
                       for v, ctx_words in _ALWAYS)
        n.threshold = n.pct and abs(value - FLAG_THRESHOLD_PCT) < 1e-9
        return n

    def _pair(self, nums: list[_Num]) -> list[_Item]:
        t = self.text
        items: list[_Item] = []
        i = 0
        while i < len(nums):
            a = nums[i]
            if i + 1 < len(nums):
                b = nums[i + 1]
                sep = t[a.end:b.digits]
                between = re.search(r"\b(?:between|tussen)\s*$", t[:a.start], re.I)
                if re.fullmatch(rf"\s*{_RANGE_JOIN}\s*", sep, re.I) or (
                        between and re.fullmatch(r"\s*(?:and|en|&)\s*", sep, re.I)):
                    b.sign = 0 if b.start < b.digits else b.sign  # the dash was a joiner
                    a.pct = a.pct or b.pct
                    items.append(_Item("range", [a, b]))
                    i += 2
                    continue
                if re.fullmatch(r"\s*(?:out of|in|op|op de|van de|of the|of|uit)\s*", sep, re.I) \
                        and 0 < a.value < b.value:
                    items.append(_Item("fraction", [a, b]))
                    i += 2
                    continue
            items.append(_Item("single", [a]))
            i += 1
        return items


def numbers_in(text: str, ctx: _Ctx | None = None) -> list[float]:
    """Every number written in ``text`` (unsigned), ignoring dates, clock times, week numbers,
    list markers and ordinals. Years are ignored only inside a date, or when ``ctx`` knows the
    year from the run's tool outputs."""
    if not text:
        return []
    return [n.value for n in _Scanner(text, ctx or _empty_ctx()).nums]


# =========================================================================== clauses

def _boundaries(text: str, pattern: str, taken: list[tuple[int, int]], lo: int, hi: int) -> list[int]:
    out = []
    for m in re.finditer(pattern, text[lo:hi]):
        p = lo + m.start()
        if all(not (a <= p < b) for a, b in taken):
            out.append(lo + m.end())
    return out


def _units(text: str, sc: _Scanner) -> list[tuple[int, int, str, set[dt.date] | None]]:
    """Split the answer into clauses (sentences, lines, semicolons), and a clause into pieces
    when a comma or 'and' separates two parts that each name a day. Returns
    (start, end, clause number, explicit dates or None, weekday-name dates or None)."""
    taken = sc.taken + [(n.start, n.end) for n in sc.nums]
    cuts = [0] + _boundaries(text, r"[.!?](?=\s|$)|[;\n]", taken, 0, len(text)) + [len(text)]
    units = []
    for ci, (s, e) in enumerate(zip(cuts, cuts[1:])):
        if not text[s:e].strip():
            continue
        sub = [s] + _boundaries(text, r",(?=\s)|\s(?:and|en|but|maar|while|whereas|terwijl)\s",
                                taken, s, e) + [e]
        pieces = []
        for ps, pe in zip(sub, sub[1:]):
            strong = {d for m in sc.dates if not m.weak and ps <= m.start < pe for d in m.dates}
            weak = {d for m in sc.dates if m.weak and ps <= m.start < pe for d in m.dates}
            pieces.append([ps, pe, strong or None, weak or None])
        # Merge pieces that name no day into their neighbour.
        groups: list[list] = []
        pending = None
        for p in pieces:
            if p[2] is None and p[3] is None:
                if groups:
                    groups[-1][1] = p[1]
                elif pending is None:
                    pending = p[0]
                continue
            if pending is not None:
                p[0], pending = pending, None
            groups.append(p)
        if not groups:
            groups = [[s, e, None, None]]
        elif pending is not None:
            groups[-1][1] = e
        groups[0][0] = s
        groups[-1][1] = e
        for g in groups:
            units.append((g[0], g[1], ci, g[2], g[3]))
    return units


# =========================================================================== faithfulness

def _is_pct_key(key: str) -> bool:
    return "pct" in key or "percent" in key


@dataclass
class _Pool:
    values: list[_Value]          # signed values from the clause's records + undated values
    question: list[float]         # unsigned numbers from the question
    pairs: list[tuple[float, float]]
    by_key: dict[str, list[tuple[dt.date, float]]]
    text: str = ""                # the whole answer (to look around a number)
    clause: str = ""              # the piece of the answer the number is in
    definition: list[float] = field(default_factory=list)  # range percentile, if allowed here
    day_counts: set[float] = field(default_factory=set)    # allowed "N days" in a flag clause


def _roles_in(text: str, known: set[str]) -> set[str]:
    words = re.findall(r"[^\W\d_]+", text.lower())
    out = set()
    for role in known:
        stems = _ROLE_STEMS.get(role, ()) + (role,)
        if any(w.startswith(s) for w in words for s in stems):
            out.add(role)
    return out


def _pool(src: _Sources, days: set[dt.date] | None, all_days: set[dt.date] | None,
          clause: str, question: list[float]) -> _Pool:
    """The values a clause may use: records of its days (only the roles it names, if any),
    undated values, and for "total"/"average" clauses the sums/means over all days of the
    sentence."""
    def records(ds):
        ds = set(src.records) if ds is None else ds & set(src.records)
        return [(d, v) for d in sorted(ds) for v in src.records[d]]

    vals = records(days)
    roles = _roles_in(clause, {v.role for _, v in vals if v.role})
    if roles:
        vals = [(d, v) for d, v in vals if v.role is None or v.role in roles]
    if days is None and _AVERAGE_RE.search(clause):
        # "On average 141 guests a day" with no day named: one day's own forecast is not an
        # average, so only the usual-for-weekday values (and undated accuracy values) count.
        vals = [(d, v) for d, v in vals if "usual" in v.key]
    values =[v for _, v in vals] + src.undated
    by_key: dict[str, list[tuple[dt.date, float]]] = {}
    for d, v in vals:
        if v.key != "derived":
            by_key.setdefault(v.key, []).append((d, v.value))
    # Totals and averages are only computed over days the sentence names: over "the next two
    # weeks" (loose) we cannot know which days the answer averaged, so such a number fails.
    if all_days is not None and (_TOTAL_RE.search(clause) or _AVERAGE_RE.search(clause)):
        groups: dict[str, list[tuple[dt.date, float]]] = {}
        for d, v in records(all_days):
            if v.key != "derived" and (not roles or v.role is None or v.role in roles):
                groups.setdefault(v.key, []).append((d, v.value))
        for key, dv in groups.items():
            if _TOTAL_RE.search(clause):    # "195 guests in total", "12 staff in total"
                values.append(_Value(sum(x for _, x in dv), key))
                for day in {d for d, _ in dv}:
                    values.append(_Value(sum(x for dd, x in dv if dd == day), key))
            if _AVERAGE_RE.search(clause):  # "97.5 guests on average"
                values.append(_Value(sum(x for _, x in dv) / len(dv), key))
    pairs = [p for d in (set(src.records) if days is None else days) for p in src.pairs.get(d, [])]
    definition = src.definition if (_DEFINITION_TEXT.search(clause)
                                    and not _COVERAGE_TEXT.search(clause)) else []
    return _Pool(values, question, pairs, by_key, clause=clause, definition=definition)


def _threshold_ok(n: _Num, pool: _Pool) -> bool:
    """Is this "15%" the flag threshold (and not a day's own change)? See _THRESHOLD_BEFORE."""
    if not n.threshold:
        return False
    before, after = pool.text[max(0, n.start - 30):n.start], pool.text[n.end:n.end + 20]
    if _THRESHOLD_BEFORE.search(before) or _THRESHOLD_AFTER.match(after):
        return True
    return not n.sign and bool(_THRESHOLD_TOPIC.search(pool.clause))


def _crosses_threshold(answer: float, source: _Value) -> bool:
    """Rounding must not move a day across the flag threshold. -14.5% may be written as
    "14.5% quieter" but not as "15% quieter": half-up rounding gives 15, which would make an
    unflagged day look flagged (and 15.0 -> "14.9" would hide a flag the other way)."""
    if not (source.dated and _is_pct_key(source.key)):
        return False
    return (abs(source.value) < FLAG_THRESHOLD_PCT) != (abs(answer) < FLAG_THRESHOLD_PCT)


def _is_day_count(n: _Num, pool: _Pool) -> bool:
    """'3 busy days', '12 days are flagged': a count of days in a clause about flags. Not
    'the next 14 days' (a window) and not the public '28 days'."""
    if n.always or n.pct or not _DAYS_AFTER.match(pool.text[n.end:n.end + 30]):
        return False
    if set(_words_before(pool.text, n.start, 1)) & _WINDOW_BEFORE:
        return False
    return any(rx.search(pool.clause) for rx in _FLAG_TOPIC.values())


def _single_ok(n: _Num, pool: _Pool) -> bool:
    if n.always or _threshold_ok(n, pool) or any(value_matches(n.value, q) for q in pool.question):
        return True
    if _is_day_count(n, pool):  # must be a count the tools returned, or the days listed
        return any(value_matches(n.value, c) for c in pool.day_counts)
    if any(value_matches(n.value, d) for d in pool.definition):
        return True
    # "10%" can only be a percentage from the tools, never "10 guests"
    cands = [v for v in pool.values if not n.pct or _is_pct_key(v.key)]
    if n.sign:
        return any(value_matches(n.sign * n.value, v.value)
                   and not _crosses_threshold(n.value, v) for v in cands)
    return any(value_matches(n.value, abs(v.value))
               and not _crosses_threshold(n.value, v) for v in cands)


def _item_ok(item: _Item, pool: _Pool) -> bool:
    if item.kind == "single":
        return _single_ok(item.nums[0], pool)
    a, b = item.nums
    if item.kind == "fraction":  # "4 out of 5 times" = 80%, "1 in 5 falls outside" = 100 - 80
        share = a.value / b.value * 100
        pcts = [abs(v.value) for v in pool.values if _is_pct_key(v.key)] + pool.definition
        if any(value_matches(share, p) or value_matches(100 - share, p) for p in pcts):
            return True
        # "x in 10", "x out of 5": a rate, so only its share counts. Otherwise ("3 of the 12
        # days") both numbers must be grounded on their own.
        if b.value <= 10 or b.value == 100:
            return False
        return _single_ok(a, pool) and _single_ok(b, pool)
    # range: a low/high pair of one record, the lowest/highest of one field over different
    # days, or two numbers from the question.
    if any(value_matches(a.value, lo) and value_matches(b.value, hi) for lo, hi in pool.pairs):
        return True
    if all(any(value_matches(n.value, q) for q in pool.question) for n in (a, b)):
        return True
    for dv in pool.by_key.values():
        da = {d for d, x in dv if value_matches(a.value, abs(x))}
        db = {d for d, x in dv if value_matches(b.value, abs(x))}
        if da and db and len(da | db) > 1:
            return True
    return False


def _resolve_units(units, src: _Sources) -> list[tuple[int, int, int, set[dt.date] | None]]:
    """Give every piece of the answer its days (None = every record, 'loose')."""
    out = []
    clause_days: set[dt.date] = set()   # days of the last clause that named or inherited any
    current_clause, current = None, set()
    for start, end, clause, strong, weak in units:
        if clause != current_clause:
            if current:
                clause_days = current
            current_clause, current = clause, set()
        if strong:                       # the clause names a date
            days = strong
        elif weak:                       # only a weekday name: prefer the days already in play
            days = (weak & (current | clause_days)) or weak
        elif clause_days:                # no day named: same days as the previous clause
            days = clause_days
        elif len(src.records) <= 1:      # nothing named yet, and the run touched one day
            days = set(src.records)
        else:                            # nothing to go on: every record, marked loose
            days = None
        current |= days or set()
        out.append((start, end, clause, days))
    return out


def faithfulness(result: RunResult) -> dict:
    """{"ungrounded": [numbers not backed by the tools for the day they talk about],
    "ok": bool, "loose": bool, "detail": [{"number", "clause", "days"}]}

    ``loose`` is True when some clause with numbers could not be tied to a day and was
    checked against every record of the run (see the module docstring)."""
    src = _collect(result)
    ctx = _ctx_for(src)
    question = numbers_in(result.question or "", ctx)
    text = result.answer or ""
    sc = _Scanner(text, ctx)
    units = _resolve_units(_units(text, sc), src)
    sentence_days: dict[int, set[dt.date] | None] = {}
    for _, _, clause, days in units:
        known = sentence_days.get(clause, set())
        sentence_days[clause] = None if days is None or known is None else known | days
    # "3 busy days" may also count the days the answer itself lists.
    listed = {d for m in sc.dates if not m.weak for d in m.dates}
    day_counts = {float(len(listed))} if listed else set()

    ungrounded: list[float] = []
    detail: list[dict] = []
    loose = False
    for start, end, clause, days in units:
        items = [it for it in sc.items if start <= it.start < end]
        if not items:
            continue
        loose = loose or days is None
        pool = _pool(src, days, sentence_days[clause], text[start:end], question)
        pool.text = text
        pool.day_counts = day_counts | set().union(*(
            c for topic, c in src.flag_counts.items() if _FLAG_TOPIC[topic].search(pool.clause)))
        for it in items:
            if _item_ok(it, pool):
                continue
            for n in it.nums:
                value = n.sign * n.value if n.sign else n.value
                ungrounded.append(value)
                detail.append({"number": value, "clause": text[start:end].strip(),
                               "days": "all (loose)" if days is None
                               else sorted(d.isoformat() for d in days)})
    return {"ungrounded": ungrounded, "ok": not ungrounded, "loose": loose, "detail": detail}


# =========================================================================== tool use

FREE_TOOLS = {"get_today_and_data_range"}  # checking today's date is never wrong
_ARG_KEYS = ("date", "start_date", "end_date", "role")
_RANGE_TOOLS = {"get_forecast", "get_flagged_days"}
# get_forecast returns the busy/quiet flag of every day, so it also answers a flagged-days call.
_ALSO_ANSWERS = {"get_flagged_days": {"get_forecast"}}


def tool_choice_ok(result: RunResult, expect: dict) -> bool:
    """All required tools were called (``tools``, plus one complete option of ``tools_any_of``),
    and nothing outside those + allowed_tools + get_today_and_data_range."""
    called = {c.name for c in result.tool_calls}
    required = set(expect.get("tools") or [])
    options = [set(o) for o in expect.get("tools_any_of") or []]
    allowed = required | set(expect.get("allowed_tools") or []) | FREE_TOOLS
    allowed |= set().union(*options) if options else set()
    if not required <= called:
        return False
    if options and not any(o <= called for o in options):
        return False
    return called <= allowed


def _norm_args(args: dict | None) -> dict:
    args = args if isinstance(args, dict) else {}
    out = {}
    for k in _ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str):
            v = v.strip().lower() or None
        out[k] = v
    return out


def _call_matches(expected: dict, actual_name: str, actual_args: dict | None) -> bool:
    if actual_name not in {expected["tool"]} | _ALSO_ANSWERS.get(expected["tool"], set()):
        return False
    want, got = _norm_args(expected.get("args")), _norm_args(actual_args)
    if want["role"] is None:  # no role asked: one call per role is fine too
        got["role"] = None
    if actual_name in _RANGE_TOOLS and want["start_date"] and got["start_date"] and got["end_date"]:
        # ISO strings compare correctly as text; a wider range that covers the asked days is fine.
        return got["start_date"] <= want["start_date"] and got["end_date"] >= want["end_date"]
    return want == got


def args_ok(result: RunResult, expect: dict) -> bool:
    """Every reference call in expect["calls"] was made by the assistant with the same
    date/role arguments (for range tools, a range covering the expected one also counts;
    a reference call without a role is matched by calls with any role)."""
    for expected in expect.get("calls") or []:
        if not any(_call_matches(expected, c.name, c.input) for c in result.tool_calls):
            return False
    return True


# =========================================================================== language

_FOOTER = re.compile(r"(?im)\b(?:based on|gebaseerd op|basiert auf|sources?|bron(?:nen)?)\s*:.*$")
_STOPWORDS = {
    "en": """the and of to you your are for with not guests guest expected expect need needs
        day days between about will staff based kitchen busier quieter tomorrow this that can
        can't cannot have there than usual on it be should more fewer likely range forecast i
        my sorry unfortunately people floor around which what saturday sunday monday tuesday
        wednesday thursday friday june july may august only data looks busy quiet today at
        from or no so but if do does would could any all bartenders usually normal""",
    "nl": """de het een en van je jij u wij zijn voor met niet gasten verwacht verwachten nodig
        er dat ook maar wordt kun kunt ongeveer tussen dag dagen drukker rustiger personeel
        keuken gebaseerd heb hebt helaas geen bij op te naar dan deze dit wat welke hoeveel
        voorspelling verwachting zaterdag zondag maandag dinsdag woensdag donderdag vrijdag
        mei augustus druk rustig normaal gemiddeld ik kan alleen vandaag weekend om tot uit
        bediening barpersoneel keukenpersoneel medewerkers waarschijnlijk minder meer
        overmorgen bezetting reserveringen januari februari maart mei juni juli oktober
        kok koks vloer nodig geboekt waarschijnlijk normaal gesloten mensen achter
        verwachten verwacht ongeveer""",
    "de": """der die das und ist nicht wir sie mit für auf ein eine einen am im zu von
        erwarten erwartet gäste gästen samstag sonntag montag dienstag mittwoch donnerstag
        freitag mai heute küche ungefähr zwischen tag tage leider keine kein nur es sind wird
        werden bei als auch oder aber wie viele prognose voraussichtlich brauchen benötigen""",
}
_LANG_WORDS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
for _a in _LANG_WORDS:  # a word that belongs to two languages says nothing
    for _b in _LANG_WORDS:
        if _a < _b:
            _shared = _LANG_WORDS[_a] & _LANG_WORDS[_b]
            _LANG_WORDS[_a] -= _shared
            _LANG_WORDS[_b] -= _shared
MIN_LANGUAGE_WORDS = 2

# Dates are the most common words in a short answer ("Vrijdag 16 juni: 4 koks."), so they
# break a tie: Dutch month/weekday names and no English ones -> Dutch (and the other way).
_DATE_WORDS = {
    "en": set("january february march june july august october monday tuesday wednesday "
              "thursday friday saturday sunday".split()),
    "nl": set("januari februari maart mei juni juli augustus oktober maandag dinsdag woensdag "
              "donderdag vrijdag zaterdag zondag".split()),
}
# Text that comes from the tools in English whatever the answer's language: role names and
# the staffing rule ("1 per 35 guests (on the forecast), rounded up, minimum 2").
_TOOL_ENGLISH = re.compile(
    r"\d+\s+per\s+\d+\s+guests?(?:\s*\(on the forecast\))?(?:,?\s*rounded up)?"
    r"(?:,?\s*minimum\s+\d+)?|\b(?:floor|bar|kitchen)\b", re.I)


def detect_language(text: str) -> str:
    """'en' | 'nl' | 'de' | 'unknown'. Counts common words of each language, ignoring the
    "Based on: ..." footer, tool names and English text copied from the tools (role names,
    staffing rules). The winner must have at least MIN_LANGUAGE_WORDS hits and strictly more
    than every other language; otherwise month/weekday names decide between en and nl."""
    text = _FOOTER.sub(" ", text or "")
    text = re.sub(r"\bget_\w+", " ", text)
    text = _TOOL_ENGLISH.sub(" ", text)
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower().replace("’", "'"))
    counts = {lang: sum(w in s for w in words) for lang, s in _LANG_WORDS.items()}
    best = max(counts, key=counts.get)
    if counts[best] >= MIN_LANGUAGE_WORDS and list(counts.values()).count(counts[best]) == 1:
        return best
    dates = {lang: any(w in s for w in words) for lang, s in _DATE_WORDS.items()}
    if dates["nl"] != dates["en"] and counts["de"] == 0:
        return "nl" if dates["nl"] else "en"
    return "unknown"


def language_ok(result: RunResult, lang: str) -> bool:
    """The answer is detectably in ``lang`` ("unknown" fails)."""
    return detect_language(result.answer) == lang


# =========================================================================== refusals

_DECLINE = {
    "en": r"can't|cannot|can not|\bunable\b|not able to|\bcan only\b|\bi only\b|don't have|"
          r"do not have|doesn't have|does not have|\bno (?:access|data|information)\b|"
          r"not available|isn't available|not possible|\bsorry\b|unfortunately|out of scope|"
          r"outside (?:the|my|what)|beyond|not something i|only (?:looks?|goes|reaches) ",
    "nl": r"kan (?:ik )?niet|kan ik geen|kan geen|kan alleen|kun je niet|niet mogelijk|"
          r"niet beschikbaar|\bhelaas\b|\bsorry\b|geen (?:toegang|gegevens|data|informatie)|"
          r"heb (?:ik )?geen|valt buiten|buiten (?:de|het|mijn)|verder dan|niet in staat|"
          r"alleen vooruit",
}


def refusal_ok(result: RunResult, expect: dict, lang: str | None = None) -> bool:
    """For must_refuse items: the answer declines in its own language, invents no numbers,
    and states no number from the forecast/staffing tools (unless ``may_share_forecast``,
    e.g. pricing questions where quoting the real forecast helps). True for in-scope items."""
    if not expect.get("must_refuse"):
        return True
    text = (result.answer or "").lower().replace("’", "'")
    if not text.strip() or not faithfulness(result)["ok"]:
        return False
    answer_lang = detect_language(result.answer)
    decline = _DECLINE.get(answer_lang) or _DECLINE.get(lang or "en", _DECLINE["en"])
    if not re.search(decline, text):
        return False
    if not expect.get("may_share_forecast") and forecast_leaks(result):
        return False
    return True


def forecast_leaks(result: RunResult) -> list[float]:
    """Numbers in the answer that are values from forecast/staffing tool outputs (numbers
    repeated from the question and the public 28 days / 15% facts do not count)."""
    src = _collect(result)
    ctx = _ctx_for(src)
    question = numbers_in(result.question or "", ctx)
    leaks = []
    for n in _Scanner(result.answer or "", ctx).nums:
        if n.always or n.threshold or any(value_matches(n.value, q) for q in question):
            continue
        if any(value_matches(n.value, abs(s)) for s in src.forecast_values):
            leaks.append(n.value)
    return leaks


# =========================================================================== expected numbers

def _values_for(obj: Any, field: str) -> list[float]:
    out: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == field and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            else:
                out.extend(_values_for(v, field))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_values_for(v, field))
    return out


def contains_expected_numbers(result: RunResult, reference_results: list[dict],
                              numbers_from: list[str] | None) -> dict:
    """Does the answer contain every value of the ``numbers_from`` fields in the reference
    results (computed at eval time on the same backend)? Same rounding as faithfulness
    (8.6 may be written as 9). {"missing": [...], "ok": bool}"""
    if not numbers_from:
        return {"missing": [], "ok": True}
    if any("error" in r for r in reference_results):
        return {"missing": [], "ok": False, "reference_error": True}
    answer_numbers = numbers_in(result.answer, _ctx_for(_collect(result)))
    missing = []
    for field_name in numbers_from:
        for value in _values_for(reference_results, field_name):
            if not any(value_matches(a, abs(value)) for a in answer_numbers) and value not in missing:
                missing.append(value)
    return {"missing": missing, "ok": not missing}


# =========================================================================== LLM judge

JUDGE_RUBRIC = """You grade one answer from a planning assistant for a restaurant manager.
Score 1-5:
5 = clear and short, gives the likely range (not only one number), says what to do
    (e.g. how many staff), and is honest about limits or uncertainty.
4 = good, one small gap (e.g. range missing or no next step).
3 = usable but vague, wordy or missing two of the points above.
2 = confusing, or overconfident about something the tools cannot know.
1 = wrong, invented, or does not answer.
For out-of-scope questions, a short polite refusal that says what it CAN answer is a 5.
Reply with the digit only."""


def judge(result: RunResult, client: Any, model: str) -> int | None:
    """LLM-as-judge score 1..5 (None if the reply has no digit). Only used with --judge."""
    tools = ", ".join(c.name for c in result.tool_calls) or "none"
    msg = client.messages.create(
        model=model,
        max_tokens=5,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content":
                   f"Question: {result.question}\nTools used: {tools}\n\nAnswer:\n{result.answer}"}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None
