"""Keep a record of every assistant run and what it cost.

One JSON line per run: easy to append, easy to load later with pandas for cost and
quality reviews.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

from agent.contract import RunResult

# US dollars per million tokens: (input, output).
# List prices Nov 2026; the Haiku price applies to prompts under 100k tokens.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


def log_run(result: RunResult, path: str | Path = "logs/runs.jsonl") -> None:
    """Append one run to the log, so cost and failures can be reviewed after the fact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
              **dataclasses.asdict(result)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Price of one run. None for unknown models: better no number than a wrong one."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
