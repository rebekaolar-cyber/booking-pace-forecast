"""The contract between the AI assistant and the forecast tool.

Every part of the agent layer (loop, evals, MCP server, Streamlit chat tab) talks to the
forecast only through ``run_tool(backend, name, args)``. A backend is anything that implements
``ForecastBackend``: ``agent.fake_backend.FakeBackend`` for tests, and the real backend that wraps the
booking-pace-forecast core.

Rules every backend and tool result follows:
* Read-only. Nothing here changes data.
* Dates are ISO strings (YYYY-MM-DD). Numbers carry their unit in the key name
  (``forecast_guests``, ``vs_usual_pct``) so the model cannot mix them up.
* Problems come back as ``{"error": "<plain sentence>"}``, never as an exception, so the
  model can explain them to the user.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_HORIZON_DAYS = 28
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FLAG_THRESHOLD_PCT = 15  # days this much busier or quieter than usual are flagged


class ForecastBackend(Protocol):
    def today_and_data_range(self) -> dict:
        """{"today", "first_booking_date", "last_booking_date", "dataset", "max_horizon_days"}"""

    def forecast(self, start_date: dt.date, end_date: dt.date) -> dict:
        """{"days": [{"date", "weekday", "booked_so_far_guests", "usual_for_weekday_guests",
        "forecast_guests", "range_low_guests", "range_high_guests", "vs_usual_pct",
        "flag" ("busy" | "quiet" | None), "closed": False}]}
        A day the place is closed is {"date", "weekday", "closed": True} with no numbers."""

    def staffing(self, date: dt.date, role: str | None) -> dict:
        """{"date", "forecast_guests", "staff": [{"role", "count", "rule"}]}"""

    def roles(self) -> list[str]:
        """Role names the staffing rules know, e.g. ["bar", "kitchen", "floor"]."""

    def accuracy(self) -> dict:
        """{"backtest": {"weeks", "description", "by_days_ahead": [{"days_ahead",
        "mean_abs_error_guests", "weekday_average_mean_abs_error_guests", "range_half_width_guests"}]},
        "range_percentile", "range_meaning",
        "after_export" (only when real outcomes are known): {"description", "days_checked",
        "inside_range_pct", "mean_abs_error_guests", "weekday_average_mean_abs_error_guests"}}"""


# --------------------------------------------------------------------------- tool schemas
# Anthropic tool-use format. Names and descriptions are what the model chooses tools by.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_today_and_data_range",
        "description": (
            "Today's date as the forecast sees it, the first and last booking dates in the data, "
            "the dataset name and the furthest day ahead that can be forecast. Call this first "
            "whenever the question uses a relative date such as 'tomorrow', 'next Saturday', "
            "'this weekend' or 'in 10 days'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_forecast",
        "description": (
            "Guest forecast per day for a date range of at most 28 days, starting today or later. "
            "For each day: guests already booked, the usual number for that weekday, the forecast, "
            "a likely range (low-high) and whether the day is flagged busy or quiet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "First day, YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "Last day, YYYY-MM-DD (inclusive)"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_staffing",
        "description": (
            "Staff needed per role on one day, from the user's own staffing rules applied to the "
            "forecast. Optionally only one role. Also returns the guest forecast it used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The day, YYYY-MM-DD"},
                "role": {"type": "string", "description": "Optional role name, e.g. 'bar'"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "get_flagged_days",
        "description": (
            f"Days in a date range (at most 28 days) forecast to be at least {FLAG_THRESHOLD_PCT}% "
            "busier or quieter than usual for their weekday."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "First day, YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "Last day, YYYY-MM-DD (inclusive)"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_accuracy",
        "description": (
            "How far off the forecast usually is, and which days each number covers. "
            "'backtest': replaying the 8 weeks before the export, the average error in guests "
            "per number of days ahead (1, 3, 7, 14), next to the weekday average's error and the "
            "likely range's half-width. 'range_percentile' 80: 8 in 10 past days stayed within that "
            "distance of the forecast. 'after_export' (only when known, e.g. the sample is history): "
            "what really happened on the next 14 days, how many landed inside the likely range "
            "and the average error."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]


# --------------------------------------------------------------------------- validation + dispatch

def _parse_date(value: Any, field_name: str) -> dt.date | str:
    if not isinstance(value, str):
        return f"{field_name} must be a date written as YYYY-MM-DD."
    if not _ISO_DATE.match(value.strip()):
        return f"{field_name} '{value}' is not a valid date; use YYYY-MM-DD."
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        return f"{field_name} '{value}' is not a valid date; use YYYY-MM-DD."


def _check_range(backend: ForecastBackend, start: dt.date, end: dt.date) -> str | None:
    info = backend.today_and_data_range()
    today = dt.date.fromisoformat(info["today"])
    last = today + dt.timedelta(days=MAX_HORIZON_DAYS - 1)
    if end < start:
        return "end_date is before start_date."
    if start < today:
        return (f"start_date {start} is in the past; the forecast covers {today} to {last}. "
                "For past days, say the forecast tool only looks ahead.")
    if end > last:
        return f"The forecast only reaches {MAX_HORIZON_DAYS} days ahead, up to {last}."
    return None


def run_tool(backend: ForecastBackend, name: str, args: dict | None) -> dict:
    """Run one tool by name. Always returns a JSON-able dict; problems come back as {"error"}."""
    args = args or {}
    try:
        if name == "get_today_and_data_range":
            # Add the weekday here, once, so the model never has to work it out itself:
            # "next Saturday" is the most common source of wrong dates in small models.
            info = dict(backend.today_and_data_range())
            today = dt.date.fromisoformat(info["today"])
            info["today_weekday"] = today.strftime("%A")
            info["last_forecast_date"] = (today + dt.timedelta(days=MAX_HORIZON_DAYS - 1)).isoformat()
            return info

        if name == "get_accuracy":
            return backend.accuracy()

        if name in ("get_forecast", "get_flagged_days"):
            start = _parse_date(args.get("start_date"), "start_date")
            end = _parse_date(args.get("end_date"), "end_date")
            for v in (start, end):
                if isinstance(v, str):
                    return {"error": v}
            problem = _check_range(backend, start, end)
            if problem:
                return {"error": problem}
            result = backend.forecast(start, end)
            if name == "get_forecast":
                return result
            return {"days": [d for d in result["days"] if d.get("flag") is not None],
                    "threshold_pct": FLAG_THRESHOLD_PCT}

        if name == "get_staffing":
            day = _parse_date(args.get("date"), "date")
            if isinstance(day, str):
                return {"error": day}
            problem = _check_range(backend, day, day)
            if problem:
                return {"error": problem}
            role = args.get("role")
            role = str(role).strip().lower() if role is not None else ""
            role = role or None  # an empty role means "all roles"
            if role is not None:
                known = backend.roles()
                if role not in known:
                    return {"error": f"Unknown role '{role}'. Known roles: {', '.join(known)}."}
            return backend.staffing(day, role)

        return {"error": f"Unknown tool '{name}'."}
    except Exception as exc:  # a tool must never crash the agent
        return {"error": f"The tool failed: {type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- run record

@dataclass
class ToolCall:
    name: str
    input: dict
    output: dict
    seconds: float


@dataclass
class RunResult:
    """What one question to the assistant produced. The loop returns this; evals and the
    run log consume it."""
    question: str
    answer: str
    model: str
    prompt_version: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str = ""  # "answered" | "tool_limit" | "error"
    error: str | None = None
