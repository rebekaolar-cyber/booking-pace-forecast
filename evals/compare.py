"""Compare two or more eval result files as a markdown table (for the README).

    python -m evals.compare evals/results/A.json evals/results/B.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = ["faithfulness", "tool_choice", "args", "refusal", "language", "numbers"]


def _pct(rate) -> str:
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def _money(value) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def row(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg, s = data["config"], data["summary"]
    cells = [f"{cfg['model']} / {cfg['prompt']} / {cfg['backend']}", str(s["questions"]),
             _pct(s["pass_rate"])]
    cells += [_pct(s["metrics"][m]["rate"]) for m in COLUMNS]
    cells += ["n/a" if s.get("judge_mean") is None else f"{s['judge_mean']:.2f}",
              f"{s['p50_seconds']} / {s['p95_seconds']}",
              f"{s['mean_input_tokens']:.0f} / {s['mean_output_tokens']:.0f}",
              _money(s["cost_per_100_questions_usd"])]
    return cells


def compare_table(paths: list[str | Path]) -> str:
    header = (["Run (model / prompt / backend)", "n", "Pass"] + [c.replace("_", " ") for c in COLUMNS]
              + ["Judge (1-5)", "p50 / p95 s", "Tokens in / out", "Cost per 100 q"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row(p)) + " |" for p in paths]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if len(paths) < 2:
        print("usage: python -m evals.compare RESULT.json RESULT.json [...]", file=sys.stderr)
        return 2
    print(compare_table(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
