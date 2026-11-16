"""MCP server: the same five forecast tools, usable from any MCP client (e.g. Claude Desktop).

This file is a thin wrapper. Every tool calls ``run_tool(BACKEND, name, args)`` from
``agent/contract.py``; validation, error messages and dispatch all live there.

Run over stdio:   python -m agent.mcp_server
Pick the backend: BPF_BACKEND=fake | real   (default: real)

The server never swaps in made-up data on its own: if the real backend can't load, every
tool returns an error saying so. With BPF_BACKEND=fake every result is marked
``"demo_data": true`` so no client can mistake it for a real forecast.
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from agent.contract import TOOL_SCHEMAS, ForecastBackend, run_tool
from agent.fake_backend import FakeBackend

# Descriptions come straight from the contract, so the MCP tools and the agent's tools match.
_SCHEMA = {t["name"]: t for t in TOOL_SCHEMAS}


def _desc(tool: str) -> str:
    return _SCHEMA[tool]["description"]


def _param(tool: str, param: str) -> str:
    return _SCHEMA[tool]["input_schema"]["properties"][param]["description"]


def make_backend() -> tuple[ForecastBackend | None, str | None]:
    """Choose the backend from BPF_BACKEND. Returns (backend, None) or (None, the reason)."""
    choice = os.environ.get("BPF_BACKEND", "real").strip().lower()
    if choice == "fake":
        return FakeBackend(), None
    if choice != "real":
        return None, f"BPF_BACKEND must be 'real' or 'fake', not '{choice}'"
    try:
        from agent.real_backend import RealBackend  # imported lazily: it is heavier
        return RealBackend(), None
    except Exception as exc:  # missing module, missing data, ...
        reason = f"the real backend could not load ({type(exc).__name__}: {exc})"
        # stderr only: on stdio, stdout belongs to the MCP protocol.
        print(f"[booking-pace-forecast] ERROR: {reason}", file=sys.stderr)
        return None, reason


BACKEND, BACKEND_PROBLEM = make_backend()


def _call(name: str, args: dict) -> dict[str, Any]:
    """Run one contract tool, refusing clearly when there is no backend and labelling demo data."""
    if BACKEND is None:
        return {"error": f"The forecast backend isn't set up: {BACKEND_PROBLEM}. "
                         "Start with BPF_BACKEND=fake to try the demo data."}
    result = run_tool(BACKEND, name, args)
    if isinstance(BACKEND, FakeBackend):
        result = {**result, "demo_data": True}  # made-up numbers: say so in every result
    return result


mcp = MCPServer("booking-pace-forecast")


@mcp.tool(description=_desc("get_today_and_data_range"))
def get_today_and_data_range() -> dict[str, Any]:
    return _call("get_today_and_data_range", {})


@mcp.tool(description=_desc("get_forecast"))
def get_forecast(
    start_date: Annotated[str, Field(description=_param("get_forecast", "start_date"))],
    end_date: Annotated[str, Field(description=_param("get_forecast", "end_date"))],
) -> dict[str, Any]:
    return _call("get_forecast", {"start_date": start_date, "end_date": end_date})


@mcp.tool(description=_desc("get_staffing"))
def get_staffing(
    date: Annotated[str, Field(description=_param("get_staffing", "date"))],
    role: Annotated[str | None, Field(description=_param("get_staffing", "role"))] = None,
) -> dict[str, Any]:
    args = {"date": date} if role is None else {"date": date, "role": role}
    return _call("get_staffing", args)


@mcp.tool(description=_desc("get_flagged_days"))
def get_flagged_days(
    start_date: Annotated[str, Field(description=_param("get_flagged_days", "start_date"))],
    end_date: Annotated[str, Field(description=_param("get_flagged_days", "end_date"))],
) -> dict[str, Any]:
    return _call("get_flagged_days", {"start_date": start_date, "end_date": end_date})


@mcp.tool(description=_desc("get_accuracy"))
def get_accuracy() -> dict[str, Any]:
    return _call("get_accuracy", {})


if __name__ == "__main__":
    mcp.run()  # stdio by default
