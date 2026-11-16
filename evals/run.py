"""Run the eval: every question through the real agent loop, scored, saved as JSON.

    python -m evals.run                         # fake backend, haiku, prompt v1
    python -m evals.run --backend real --sample 10   # 10 questions, spread over categories
    python -m evals.run --ids sd01 nl05 --judge

Reference numbers are computed here, at eval time, by calling the same tools on the same
backend the assistant uses. Exit code 1 if any answer contains an invented number
(faithfulness below 100%) or any run failed with an error.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.contract import RunResult, run_tool
from agent.fake_backend import FakeBackend
from evals import scoring

EVALS_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVALS_DIR / "questions.yaml"
METRICS = ["faithfulness", "tool_choice", "args", "refusal", "language", "numbers"]


# --------------------------------------------------------------------------- loading

def load_questions(path: str | Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stratified_sample(questions: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n questions spread over the categories in proportion to their size, at least one per
    category when n allows it. Fixed seed, so the same n always gives the same questions.
    Returned in file order."""
    if n >= len(questions):
        return list(questions)
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(q["category"], []).append(q)
    names = list(groups)
    base = 1 if n >= len(names) else 0
    quota = {c: base for c in names}
    sizes = {c: len(groups[c]) - base for c in names}
    remaining = n - base * len(names)
    shares = {c: remaining * sizes[c] / sum(sizes.values()) for c in names}
    for c in names:
        quota[c] += math.floor(shares[c])
    left = n - sum(quota.values())
    by_remainder = sorted(names, key=lambda c: (-(shares[c] - math.floor(shares[c])), names.index(c)))
    for c in by_remainder[:left]:
        quota[c] += 1
    rng = random.Random(seed)
    chosen = {q["id"] for c in names for q in rng.sample(groups[c], quota[c])}
    return [q for q in questions if q["id"] in chosen]


def select(questions: list[dict], ids: list[str] | None = None, limit: int | None = None,
           sample: int | None = None, seed: int = 0) -> list[dict]:
    if ids:
        unknown = set(ids) - {q["id"] for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in ids]
    if sample is not None:
        questions = stratified_sample(questions, sample, seed)
    if limit is not None:
        questions = questions[:limit]
    return questions


def make_backend(name: str):
    if name == "fake":
        return FakeBackend()
    if name == "real":
        try:
            from agent.real_backend import RealBackend
        except ImportError as exc:
            raise SystemExit(
                "--backend real needs agent/real_backend.py with a RealBackend class "
                f"(import failed: {exc}). Use --backend fake to run without it.")
        return RealBackend()
    raise SystemExit(f"Unknown backend '{name}' (use fake or real).")


def _load_answer() -> Callable[..., RunResult]:
    from agent.loop import answer  # lazy: tests stub this module
    return answer


def _load_cost() -> Callable[[str, int, int], float | None]:
    try:
        from agent.run_log import cost_usd
    except ImportError:
        return lambda model, i, o: None
    return cost_usd


# --------------------------------------------------------------------------- one question

def score_question(item: dict, result: RunResult, reference: list[dict]) -> dict:
    """All automatic checks for one question. None = check does not apply to this item."""
    expect = item["expect"]
    faith = scoring.faithfulness(result)
    numbers = scoring.contains_expected_numbers(result, reference, expect.get("numbers_from"))
    scores = {
        "faithfulness": faith["ok"],
        "tool_choice": scoring.tool_choice_ok(result, expect),
        "args": scoring.args_ok(result, expect),
        "refusal": (scoring.refusal_ok(result, expect, item.get("lang", "en"))
                    if expect.get("must_refuse") else None),
        "language": scoring.language_ok(result, item.get("lang", "en")),
        "numbers": numbers["ok"] if expect.get("numbers_from") else None,
    }
    passed = result.error is None and all(v for v in scores.values() if v is not None)
    return {"scores": scores, "passed": passed, "ungrounded": faith["ungrounded"],
            "ungrounded_detail": faith["detail"],
            # True = some clause named no day, so its numbers were checked against every day
            "faithfulness_loose": faith["loose"],
            "missing_numbers": numbers["missing"],
            "reference_error": bool(numbers.get("reference_error"))}


def pick_reading(item: dict, result: RunResult) -> dict:
    """Some questions have two fair readings ("next Saturday" said on a Thursday). The item
    lists the second one as ``alt_calls``; if the assistant clearly used it, score against it."""
    alt = item["expect"].get("alt_calls")
    if not alt:
        return item
    main_ok = scoring.args_ok(result, item["expect"])
    alt_expect = {**item["expect"], "calls": alt}
    if not main_ok and scoring.args_ok(result, alt_expect):
        return {**item, "expect": alt_expect}
    return item


def print_failures(records: list[dict]) -> None:
    """Show every failed question in full, so a CI log alone is enough to judge whether the
    model or the scorer was wrong (the results file is also uploaded as an artifact)."""
    failed = [r for r in records if not r["passed"]]
    if not failed:
        return
    print("\nFailed questions:")
    for r in failed:
        checks = [name for name, ok in r["scores"].items() if ok is False]
        print(f"\n--- {r['id']} ({r['category']}, {r['lang']}) failed: {', '.join(checks) or r['stop_reason']}")
        print(f"Q: {r['question']}")
        print("A: " + (r["answer"] or "").replace("\n", "\n   "))
        for c in r["tool_calls"]:
            print(f"   tool {c['name']} {json.dumps(c['input'], ensure_ascii=False)}")
        if r["ungrounded"]:
            print(f"   not traced to a tool: {r['ungrounded']}")
            for d in r.get("ungrounded_detail") or []:
                print(f"     {d}")
        if r.get("missing_numbers"):
            print(f"   expected but missing: {r['missing_numbers']}")
        if r.get("note"):
            print(f"   note: {r['note']}")


def run_one(item: dict, backend, answer_fn: Callable[..., RunResult], *, model: str,
            prompt: str, cost_fn: Callable, judge_client=None, judge_model: str | None = None) -> dict:
    started = time.perf_counter()
    result = answer_fn(item["question"], backend, model=model, prompt_version=prompt)
    wall = time.perf_counter() - started
    item = pick_reading(item, result)
    reference = [run_tool(backend, c["tool"], c.get("args") or {})
                 for c in item["expect"].get("calls") or []]

    record = {
        "id": item["id"], "category": item["category"], "lang": item.get("lang", "en"),
        "question": item["question"], "note": item["expect"].get("note"),
        "answer": result.answer, "stop_reason": result.stop_reason, "error": result.error,
        "tool_calls": [dataclasses.asdict(c) for c in result.tool_calls],
        "reference": [{"tool": c["tool"], "args": c.get("args") or {}, "output": out}
                      for c, out in zip(item["expect"].get("calls") or [], reference)],
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "seconds": result.seconds or round(wall, 3),
        "cost_usd": cost_fn(result.model or model, result.input_tokens, result.output_tokens),
        **score_question(item, result, reference),
        "judge": None,
    }
    if judge_client is not None:
        try:
            record["judge"] = scoring.judge(result, judge_client, judge_model or model)
        except Exception as exc:  # a judge failure must not hide the automatic scores
            record["judge_error"] = str(exc)
    return record


# --------------------------------------------------------------------------- summary

def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile: simple and easy to check by hand."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    metrics = {}
    for m in METRICS:
        applicable = [r["scores"][m] for r in records if r["scores"][m] is not None]
        metrics[m] = {"passed": sum(applicable), "applicable": len(applicable),
                      "rate": round(sum(applicable) / len(applicable), 4) if applicable else None}
    by_category: dict[str, dict] = {}
    for r in records:
        c = by_category.setdefault(r["category"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += int(r["passed"])
    seconds = [r["seconds"] for r in records]
    costs = [r["cost_usd"] for r in records]
    total_cost = None if any(c is None for c in costs) or not costs else sum(costs)
    judged = [r["judge"] for r in records if r["judge"] is not None]
    return {
        "questions": n,
        "passed": sum(r["passed"] for r in records),
        "pass_rate": round(sum(r["passed"] for r in records) / n, 4) if n else None,
        "metrics": metrics,
        "by_category": by_category,
        "errors": sum(1 for r in records if r["error"]),
        "faithfulness_loose": sum(1 for r in records if r.get("faithfulness_loose")),
        "p50_seconds": percentile(seconds, 50),
        "p95_seconds": percentile(seconds, 95),
        "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in records), 1) if n else None,
        "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in records), 1) if n else None,
        "total_cost_usd": round(total_cost, 6) if total_cost is not None else None,
        "cost_per_100_questions_usd": round(total_cost / n * 100, 4) if total_cost is not None and n else None,
        "judge_mean": round(statistics.mean(judged), 2) if judged else None,
        "judged": len(judged),
    }


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def format_summary(summary: dict, config: dict) -> str:
    lines = [f"Eval: {config['model']} / prompt {config['prompt']} / {config['backend']} backend "
             f"/ {summary['questions']} questions",
             f"  passed (all checks)  {summary['passed']}/{summary['questions']}  "
             f"({_pct(summary['pass_rate'])})"]
    for m, s in summary["metrics"].items():
        lines.append(f"  {m:<20} {s['passed']}/{s['applicable']}  ({_pct(s['rate'])})")
    if summary["judge_mean"] is not None:
        lines.append(f"  judge mean (1-5)     {summary['judge_mean']}  ({summary['judged']} judged)")
    lines.append(f"  loose faithfulness   {summary['faithfulness_loose']} (answers that name no day; "
                 "checked against every day)")
    lines.append(f"  errors               {summary['errors']}")
    lines.append(f"  latency              p50 {summary['p50_seconds']} s, p95 {summary['p95_seconds']} s")
    lines.append(f"  mean tokens          {summary['mean_input_tokens']} in, "
                 f"{summary['mean_output_tokens']} out")
    if summary["total_cost_usd"] is None:
        lines.append("  cost                 unknown (no price for this model)")
    else:
        lines.append(f"  cost                 ${summary['total_cost_usd']:.4f} total, "
                     f"${summary['cost_per_100_questions_usd']:.4f} per 100 questions")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main

def run_eval(questions: list[dict], backend, answer_fn: Callable[..., RunResult], *,
             model: str, prompt: str, cost_fn: Callable | None = None, judge_client=None,
             judge_model: str | None = None, progress: bool = True) -> list[dict]:
    cost_fn = cost_fn or (lambda m, i, o: None)
    records = []
    for i, item in enumerate(questions, 1):
        record = run_one(item, backend, answer_fn, model=model, prompt=prompt, cost_fn=cost_fn,
                         judge_client=judge_client, judge_model=judge_model)
        records.append(record)
        if progress:
            mark = "PASS" if record["passed"] else "FAIL"
            failed = [m for m, v in record["scores"].items() if v is False]
            extra = f"  failed: {', '.join(failed)}" if failed else ""
            print(f"[{i}/{len(questions)}] {mark} {item['id']}{extra}", flush=True)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["fake", "real"], default="fake")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--prompt", default="v1", help="prompt version, prompts/<version>.md")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--sample", type=int, default=None,
                   help="N questions spread over the categories (fixed seed)")
    p.add_argument("--seed", type=int, default=0, help="seed for --sample")
    p.add_argument("--ids", nargs="+", default=None, help="only these question ids")
    p.add_argument("--judge", action="store_true", help="also score answers 1-5 with an LLM judge")
    p.add_argument("--judge-model", default=None, help="model for --judge (default: --model)")
    p.add_argument("--questions", default=str(QUESTIONS_PATH))
    p.add_argument("--out", default=str(EVALS_DIR / "results"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = select(load_questions(args.questions), args.ids, args.limit, args.sample, args.seed)
    backend = make_backend(args.backend)
    answer_fn = _load_answer()
    judge_client = None
    if args.judge:
        import anthropic
        judge_client = anthropic.Anthropic()

    config = {"backend": args.backend, "model": args.model, "prompt": args.prompt,
              "judge_model": (args.judge_model or args.model) if args.judge else None,
              "questions_file": Path(args.questions).name, "ids": args.ids, "limit": args.limit,
              "sample": args.sample, "seed": args.seed if args.sample is not None else None}
    started = dt.datetime.now(dt.timezone.utc)
    records = run_eval(questions, backend, answer_fn, model=args.model, prompt=args.prompt,
                       cost_fn=_load_cost(), judge_client=judge_client,
                       judge_model=args.judge_model)
    summary = summarize(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{stamp}_{args.model}_{args.prompt}.json"
    payload: dict[str, Any] = {"started_utc": started.isoformat(), "config": config,
                               "summary": summary, "results": records}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(format_summary(summary, config))
    print(f"\nSaved {out_path}")
    print_failures(records)

    faith = summary["metrics"]["faithfulness"]["rate"]
    if faith is not None and faith < 1.0:
        bad = [f"{r['id']} {r['ungrounded']}" for r in records if r["ungrounded"]]
        print("FAIL: invented numbers in " + "; ".join(bad))
        return 1
    if summary["errors"]:
        print(f"FAIL: {summary['errors']} run(s) ended in an error")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
