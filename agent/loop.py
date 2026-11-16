"""The agent loop: one question in, one RunResult out.

A plain tool-use loop on the Anthropic Messages API, no framework, so every step can be
read and explained: ask the model, run the tools it asks for, send the results back,
repeat until it answers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import anthropic

from agent.contract import TOOL_SCHEMAS, RunResult, ToolCall, run_tool

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
MAX_HISTORY_TURNS = 4  # earlier question/answer pairs sent along, so "and Sunday?" works

TOO_MANY_STEPS = ("Sorry, that question needed too many steps for me to answer reliably. "
                  "Could you ask it in a simpler or narrower way, for example one day at a time?")
NO_API_KEY = "The assistant isn't set up yet: no Anthropic API key was found."
API_PROBLEM = "Sorry, I couldn't reach the AI service just now. Please try again in a moment."
INTERNAL_PROBLEM = ("Sorry, something went wrong inside the assistant (an internal error, "
                    "not the AI service). The problem has been logged.")
NO_TEXT = "Sorry, I couldn't put an answer together. Please try rephrasing the question."
REFUSED = "Sorry, I can't help with that question."
CUT_OFF = "(answer cut off)"
BUDGET_REACHED = {"error": "tool budget reached"}


def load_prompt(version: str) -> str:
    """Prompts live in versioned files so evals can compare v1 against v2 without code changes."""
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


def block_to_dict(block: Any) -> dict:
    """Turn a response content block into the plain dict the API accepts back in `messages`.

    Real SDK blocks are pydantic models (model_dump); test fakes are simple namespaces.
    """
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    raise ValueError(f"Unsupported content block type: {block.type}")


def history_messages(history: list[dict] | None) -> list[dict]:
    """Earlier turns as plain user/assistant text (only the last few, to keep cost down)."""
    messages = []
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        question, reply = turn.get("question"), turn.get("answer")
        if question and reply:  # the API rejects empty text
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": reply})
    return messages


def answer(question: str, backend, client=None, model: str = "claude-haiku-5-5",
           prompt_version: str = "v2", history: list[dict] | None = None,
           max_rounds: int = 6, max_tool_calls: int = 12,
           max_tokens: int = 700) -> RunResult:
    """Answer one question with the forecast tools. Never raises: problems become an
    'error' RunResult so the chat page and the evals can always show something.

    Two budgets stop a confused model from looping (and spending) forever:
    max_rounds model calls, and max_tool_calls tool calls in total.
    """
    result = RunResult(question=question, answer="", model=model, prompt_version=prompt_version)
    system = load_prompt(prompt_version)
    messages = history_messages(history) + [{"role": "user", "content": question}]
    started = time.perf_counter()

    try:
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                # The SDK only notices a missing key at the first request, with a TypeError
                # that would look like a bug in our code; say what's really wrong instead.
                result.answer, result.stop_reason = NO_API_KEY, "error"
                result.error = "ANTHROPIC_API_KEY is not set"
                return result
            client = anthropic.Anthropic()

        for round_no in range(1, max_rounds + 1):
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                tools=TOOL_SCHEMAS, messages=messages,
            )
            usage = getattr(response, "usage", None)
            result.input_tokens += getattr(usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(usage, "output_tokens", 0) or 0

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                _finish(result, response)
                break
            if round_no == max_rounds:  # no model call left to answer with the results
                result.answer = TOO_MANY_STEPS
                result.stop_reason = "tool_limit"
                break

            messages.append({"role": "assistant",
                             "content": [block_to_dict(b) for b in response.content]})
            messages.append({"role": "user",
                             "content": _run_tools(backend, tool_uses, result, max_tool_calls)})

    # AnthropicError covers API errors and set-up problems such as a missing API key.
    except (anthropic.AnthropicError, ConnectionError, TimeoutError) as exc:  # API, key or network
        result.answer = API_PROBLEM
        result.stop_reason = "error"
        result.error = str(exc)
    except Exception as exc:  # a bug in our code: say so, don't blame the AI service
        result.answer = INTERNAL_PROBLEM
        result.stop_reason = "error"
        result.error = f"{type(exc).__name__}: {exc}"

    result.seconds = round(time.perf_counter() - started, 3)
    return result


def _finish(result: RunResult, response) -> None:
    """Record the final answer and the real reason the model stopped."""
    text = "".join(b.text for b in response.content
                   if getattr(b, "type", None) == "text").strip()
    stop = response.stop_reason
    if stop == "end_turn":
        result.stop_reason = "answered"
        result.answer = text or NO_TEXT
    elif stop == "max_tokens":
        result.stop_reason = "truncated"
        if not text.endswith((".", "!", "?")):  # empty, or stopped mid-sentence
            text = f"{text} {CUT_OFF}".strip()
        result.answer = text
    elif stop == "refusal":
        result.stop_reason = "refused"
        result.answer = text or REFUSED
    else:  # pause_turn, "tool_use" without tool blocks, anything new
        result.stop_reason = "error"
        result.answer = text or NO_TEXT
        result.error = f"Unexpected stop_reason: {stop}"


def _run_tools(backend, tool_uses: list, result: RunResult, max_tool_calls: int) -> list[dict]:
    """Run every tool the model asked for in this turn; all results go back in one message,
    as the API requires. Calls over the budget are not run: they get an error result, so the
    model can still answer with what it has."""
    tool_results = []
    for block in tool_uses:
        args = block.input if isinstance(block.input, dict) else {}  # run_tool reports what's missing
        if len(result.tool_calls) >= max_tool_calls:
            output = BUDGET_REACHED
        else:
            t0 = time.perf_counter()
            output = run_tool(backend, block.name, args)
            seconds = round(time.perf_counter() - t0, 4)
            result.tool_calls.append(ToolCall(name=block.name, input=dict(args),
                                              output=output, seconds=seconds))
        tool_result = {"type": "tool_result", "tool_use_id": block.id,
                       "content": json.dumps(output)}
        if "error" in output:
            tool_result["is_error"] = True  # tells the model the call failed, not the data
        tool_results.append(tool_result)
    return tool_results
