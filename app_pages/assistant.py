"""Streamlit chat page for the AI assistant.

The main app calls ``render(backend)``; ``demo_assistant.py`` runs it with FakeBackend.
All the AI work happens in ``agent.loop.answer``; this file is only the UI around it.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading

import streamlit as st

from agent.loop import answer
from agent.run_log import cost_usd, log_run

# Two soft caps keep a public demo from running up the API bill. The session cap resets when
# someone reloads the page, so there is also a daily cap shared by everyone on this server.
# Neither is the real protection: that is the monthly spend limit set on the API key itself.
MAX_QUESTIONS_PER_SESSION = 10
MAX_QUESTIONS_PER_DAY = 200  # all sessions together, per UTC day
EXAMPLE_QUESTIONS = [
    "How many bar staff do I need on Saturday?",
    "Any unusual days in the next two weeks?",
    "Wordt het druk aanstaande vrijdag?",
]
DAILY_LIMIT_NOTE = ("The demo has answered all the questions it can for today. "
                    "Please come back tomorrow.")
HISTORY_KEY = "assistant_history"  # prefixed so it can't clash with the main app's state


def _api_key() -> str | None:
    """Secrets file first (Streamlit Cloud), then the environment (local runs)."""
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:  # no secrets file, or no key in it
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def _client(api_key: str):
    """One client per key for the whole server, instead of one per question."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


class DailyCounter:
    """Questions asked today (UTC) by all sessions together. The lock stops two sessions
    that ask at the same moment from both taking the last question."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: dt.date | None = None
        self._count = 0

    def _reset_if_new_day(self) -> None:
        today = dt.datetime.now(dt.timezone.utc).date()
        if self._day != today:
            self._day, self._count = today, 0

    def reached(self, limit: int) -> bool:
        with self._lock:
            self._reset_if_new_day()
            return self._count >= limit

    def take(self, limit: int) -> bool:
        """Count one question; False (and nothing counted) if today's limit is already used."""
        with self._lock:
            self._reset_if_new_day()
            if self._count >= limit:
                return False
            self._count += 1
            return True


@st.cache_resource
def _daily_counter() -> DailyCounter:
    """One counter for the whole server: cache_resource objects are shared by all sessions."""
    return DailyCounter()


def _show_details(result) -> None:
    """The 'How I got this' box: lets a manager (or reviewer) check where numbers came from."""
    with st.expander("How I got this"):
        if result.tool_calls:
            for call in result.tool_calls:
                st.markdown(f"- `{call.name}` {json.dumps(call.input)} ({call.seconds:.2f} s)")
        else:
            st.markdown("No tools were used.")
        cost = cost_usd(result.model, result.input_tokens, result.output_tokens)
        cost_text = f"{cost:.5f} USD" if cost is not None else "unknown"
        st.caption(f"{result.seconds:.1f} s total · {result.input_tokens} in / "
                   f"{result.output_tokens} out tokens · cost {cost_text} · {result.model}")


def _show_turn(question: str, result) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(result.answer)
        _show_details(result)


def render(backend) -> None:
    """Draw the chat page. Safe without an API key: it explains and disables input."""
    st.subheader("Ask the assistant")
    st.caption("Questions about expected guests and staffing, in English or Dutch.")

    history: list[tuple[str, object]] = st.session_state.setdefault(HISTORY_KEY, [])
    api_key = _api_key()
    counter = _daily_counter()
    session_limit = len(history) >= MAX_QUESTIONS_PER_SESSION
    daily_limit = counter.reached(MAX_QUESTIONS_PER_DAY)

    if not api_key:
        st.info("The assistant needs an Anthropic API key. Add ANTHROPIC_API_KEY to "
                ".streamlit/secrets.toml or set it as an environment variable.")
    if session_limit:
        st.info(f"You've asked {MAX_QUESTIONS_PER_SESSION} questions, the limit for one "
                "session of this demo. Thanks for trying it!")
    elif daily_limit:
        st.info(DAILY_LIMIT_NOTE)
    disabled = not api_key or session_limit or daily_limit

    question = None
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example, disabled=disabled):
            question = example

    for past_question, past_result in history:
        _show_turn(past_question, past_result)

    typed = st.chat_input("Ask about guests or staffing...", disabled=disabled)
    question = question or typed
    if not question or disabled:
        return

    if not counter.take(MAX_QUESTIONS_PER_DAY):  # another session took the last one
        st.info(DAILY_LIMIT_NOTE)
        return

    # Earlier turns give follow-ups like "and Sunday?" their context; the loop keeps the last 4.
    earlier = [{"question": q, "answer": r.answer} for q, r in history if r.stop_reason != "error"]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the forecast..."):
            result = answer(question, backend, client=_client(api_key), history=earlier)
        log_run(result)
        st.markdown(result.answer)
        _show_details(result)
    history.append((question, result))
    # This answer may have used the last question. The input above was drawn before it, still
    # enabled, so draw the page again: it then shows the limit note and disables the input.
    if len(history) >= MAX_QUESTIONS_PER_SESSION or counter.reached(MAX_QUESTIONS_PER_DAY):
        st.rerun()
