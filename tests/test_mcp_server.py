"""The MCP server exposes the five contract tools and returns what run_tool returns, marked as
demo data with the fake backend, and refuses clearly (never falls back) without a backend."""
import asyncio
import os
import sys
from pathlib import Path

os.environ["BPF_BACKEND"] = "fake"  # before the server module picks its backend

import pytest
from mcp import Client, StdioServerParameters

from agent import mcp_server
from agent.contract import TOOL_NAMES, run_tool
from agent.fake_backend import FakeBackend

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(mcp_server, "BACKEND", FakeBackend())
    monkeypatch.setattr(mcp_server, "BACKEND_PROBLEM", None)


def _with_client(server, fn):
    async def go():
        async with Client(server) as client:
            return await fn(client)
    return asyncio.run(go())


def test_lists_the_five_contract_tools():
    async def names(client):
        return sorted(t.name for t in (await client.list_tools()).tools)
    assert _with_client(mcp_server.mcp, names) == sorted(TOOL_NAMES)


def test_get_staffing_matches_run_tool():
    async def call(client):
        return await client.call_tool("get_staffing", {"date": "2017-06-03"})
    result = _with_client(mcp_server.mcp, call)
    assert not result.is_error
    expected = run_tool(FakeBackend(), "get_staffing", {"date": "2017-06-03"})
    assert result.structured_content == {**expected, "demo_data": True}


def test_past_date_error_comes_back_as_data():
    async def call(client):
        return await client.call_tool("get_forecast", {"start_date": "2017-05-01", "end_date": "2017-05-02"})
    result = _with_client(mcp_server.mcp, call)
    assert not result.is_error  # not a protocol error: the model gets a sentence to explain
    assert "in the past" in result.structured_content["error"]
    assert result.structured_content["demo_data"] is True


def test_every_tool_is_marked_demo_data_with_the_fake_backend():
    args = {"get_today_and_data_range": {}, "get_accuracy": {},
            "get_forecast": {"start_date": "2017-06-01", "end_date": "2017-06-02"},
            "get_flagged_days": {"start_date": "2017-06-01", "end_date": "2017-06-20"},
            "get_staffing": {"date": "2017-06-03", "role": "bar"}}

    async def call_all(client):
        return {name: (await client.call_tool(name, a)).structured_content
                for name, a in args.items()}
    results = _with_client(mcp_server.mcp, call_all)
    assert sorted(results) == sorted(TOOL_NAMES)
    assert all(r["demo_data"] is True for r in results.values())


def test_no_backend_means_every_tool_returns_a_setup_error(monkeypatch):
    monkeypatch.setattr(mcp_server, "BACKEND", None)
    monkeypatch.setattr(mcp_server, "BACKEND_PROBLEM", "the real backend could not load (x)")

    async def call(client):
        return await client.call_tool("get_staffing", {"date": "2017-06-03"})
    result = _with_client(mcp_server.mcp, call).structured_content
    assert result == {"error": "The forecast backend isn't set up: the real backend could not "
                               "load (x). Start with BPF_BACKEND=fake to try the demo data."}
    for name in TOOL_NAMES:
        assert "isn't set up" in mcp_server._call(name, {})["error"]


def test_real_backend_that_cannot_load_does_not_fall_back_to_fake(monkeypatch):
    monkeypatch.setenv("BPF_BACKEND", "real")
    monkeypatch.setitem(sys.modules, "agent.real_backend", None)  # import fails
    backend, problem = mcp_server.make_backend()
    assert backend is None
    assert "real backend could not load" in problem


def test_default_backend_is_real(monkeypatch):
    monkeypatch.delenv("BPF_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "agent.real_backend", None)
    backend, problem = mcp_server.make_backend()
    assert backend is None and "real backend" in problem


def test_unknown_backend_name_is_an_error(monkeypatch):
    monkeypatch.setenv("BPF_BACKEND", "fkae")
    backend, problem = mcp_server.make_backend()
    assert backend is None
    assert "'real' or 'fake'" in problem


def test_fake_backend_choice(monkeypatch):
    monkeypatch.setenv("BPF_BACKEND", "Fake ")
    backend, problem = mcp_server.make_backend()
    assert isinstance(backend, FakeBackend) and problem is None


def test_runs_over_stdio_like_claude_desktop():
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent.mcp_server"],
                                   cwd=str(REPO), env={"BPF_BACKEND": "fake"})

    async def names(client):
        return sorted(t.name for t in (await client.list_tools()).tools)
    assert _with_client(params, names) == sorted(TOOL_NAMES)


def test_stdio_without_a_backend_refuses_instead_of_using_fake_data():
    # A misspelt name, so this holds whether or not agent/real_backend.py exists yet.
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent.mcp_server"],
                                   cwd=str(REPO), env={"BPF_BACKEND": "fkae"})

    async def call(client):
        return await client.call_tool("get_accuracy", {})
    result = _with_client(params, call).structured_content
    assert "isn't set up" in result["error"]
    assert "demo_data" not in result
