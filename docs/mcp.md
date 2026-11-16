# Using the forecast from any MCP client

MCP (Model Context Protocol) is an open standard that lets an AI app such as Claude Desktop
call tools that run on your own machine. `agent/mcp_server.py` exposes the same five tools as
the built-in assistant (`get_today_and_data_range`, `get_forecast`, `get_staffing`,
`get_flagged_days`, `get_accuracy`). Each tool just calls `run_tool` from `agent/contract.py`,
so validation and error messages are identical everywhere.

## Run the server

From the repo root:

```bash
BPF_BACKEND=fake python -m agent.mcp_server
```

It talks MCP over stdin/stdout, so it waits silently; a client starts it for you.

- `BPF_BACKEND=fake`: deterministic, made-up demo data ("today" is Thursday 2017-06-01). Every tool
  result carries `"demo_data": true`, so a client can't mistake it for a real forecast.
- `BPF_BACKEND=real` (the default): the real forecast. If `agent.real_backend` cannot be loaded,
  the server does **not** fall back to demo data: it prints the reason to stderr and every tool
  returns `{"error": "The forecast backend isn't set up: <reason>. Start with BPF_BACKEND=fake to
  try the demo data."}`, which the model can pass on to the user. Any other `BPF_BACKEND` value
  is treated the same way.

Problems (a past date, an unknown role, a range over 28 days) come back as a normal result,
`{"error": "<plain sentence>"}`, so the model can explain them.

## Claude Desktop

Add this to `claude_desktop_config.json` (Settings > Developer > Edit Config), replace the
`cwd` with the absolute path of this repo, and restart Claude Desktop. Use the full path to
the Python that has `mcp` installed if `python` on your PATH is a different one. If your client
ignores `cwd` (the error is `No module named agent`), also add `"PYTHONPATH"` with the same path to `env`.

```json
{
  "mcpServers": {
    "booking-pace-forecast": {
      "command": "python",
      "args": ["-m", "agent.mcp_server"],
      "cwd": "/absolute/path/to/booking-pace-forecast",
      "env": { "BPF_BACKEND": "fake" }
    }
  }
}
```

## Example prompts

- "What dates can you forecast, and how accurate has the forecast been?"
- "How many bar staff do I need on 2017-06-03?"
- "Which days in the next two weeks look unusually busy or quiet?"

## Tested

`python -m pytest -q tests/test_mcp_server.py` connects an MCP client to the server, both
in-process and over stdio (`python -m agent.mcp_server` with `BPF_BACKEND=fake`), and checks the
tool list, a `get_staffing` result against `run_tool`, a past-date error returned as data, the
`demo_data` marker on every tool, and the setup error (no fallback) when there is no backend.
The Claude Desktop config above was not run in this environment.
