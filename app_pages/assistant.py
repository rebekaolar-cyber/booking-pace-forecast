"""Streamlit chat page for the AI assistant.

The main app calls ``render(backend)``; ``demo_assistant.py`` runs it with FakeBackend.
All the AI work happens in ``agent.loop.answer``; this file is only the UI around it.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import os
import sys
import threading

import streamlit as st

from agent import loop
from agent.contract import ToolCall
from agent.loop import answer
from agent.run_log import cost_usd, log_run
from app_pages.i18n import tr

# Two soft caps keep a public demo from running up the API bill. The session cap resets when
# someone reloads the page, so there is also a daily cap shared by everyone on this server.
# Neither is the real protection: that is the monthly spend limit set on the API key itself.
MAX_QUESTIONS_PER_SESSION = 10
MAX_QUESTIONS_PER_DAY = 200  # all sessions together, per UTC day
EXAMPLE_QUESTIONS = [  # one Dutch question in English mode shows off the second language
    "How many bar staff do I need on Saturday?",
    "Any unusual days in the next two weeks?",
    "Wordt het druk aanstaande vrijdag?",
]
EXAMPLE_QUESTIONS_NL = [  # and one English question in Dutch mode
    "Hoeveel barpersoneel heb ik zaterdag nodig?",
    "Zijn er ongewone dagen in de komende twee weken?",
    "Will it be busy this Friday?",
]
DAILY_LIMIT_NOTE = ("The demo has answered all the questions it can for today. "
                    "Please come back tomorrow.")
DAILY_LIMIT_NOTE_NL = ("De demo heeft vandaag alle vragen beantwoord die hij kan. "
                       "Kom morgen gerust terug.")
# The loop's fixed replies (errors, limits) are English; the model's own answers already
# follow the question's language. Shown in Dutch when the page is in Dutch.
FIXED_REPLIES_NL = {
    loop.TOO_MANY_STEPS: ("Sorry, voor die vraag waren te veel stappen nodig om betrouwbaar te "
                          "antwoorden. Kun je hem eenvoudiger of smaller stellen, bijvoorbeeld per dag?"),
    loop.NO_API_KEY: "De assistent is nog niet ingesteld: er is geen Anthropic API-sleutel gevonden.",
    loop.API_PROBLEM: "Sorry, ik kon de AI-dienst nu niet bereiken. Probeer het zo nog eens.",
    loop.INTERNAL_PROBLEM: ("Sorry, er ging iets mis in de assistent zelf (een interne fout, niet "
                            "de AI-dienst). Het probleem is gelogd."),
    loop.NO_TEXT: "Sorry, ik kon geen antwoord samenstellen. Probeer de vraag anders te stellen.",
    loop.REFUSED: "Sorry, met die vraag kan ik niet helpen.",
}
HISTORY_KEY = "assistant_history"  # prefixed so it can't clash with the main app's state
AVATARS = {"user": ":material/person:", "assistant": ":material/room_service:"}


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


def _reply_text(result) -> str:
    """The answer as shown: the model's own words, or a fixed reply in the page's language."""
    return tr(result.answer, FIXED_REPLIES_NL.get(result.answer, result.answer))


# What usually lies behind an API error, in plain words: (text found in the error, English, Dutch).
# Checked in order; the first match wins.
ERROR_HINTS = [
    (r"credit balance", "The Anthropic account behind this demo has run out of credits.",
     "Het Anthropic-account achter deze demo heeft geen tegoed meer."),
    (r"error code: 401|authentication_error|invalid x-api-key",
     "The API key was rejected: it may be mistyped or deleted.",
     "De API-sleutel werd geweigerd: hij is misschien verkeerd overgenomen of verwijderd."),
    (r"error code: 403|permission_error", "The API key isn't allowed to use this model.",
     "De API-sleutel mag dit model niet gebruiken."),
    (r"error code: 404|not_found_error", "The model name wasn't found.", "De modelnaam werd niet gevonden."),
    (r"error code: 429|rate_limit_error", "Too many questions at once for this API key. Try again in a minute.",
     "Te veel vragen tegelijk voor deze API-sleutel. Probeer het over een minuut opnieuw."),
    (r"error code: 529|overloaded", "Anthropic's service is busy right now. Try again in a moment.",
     "De dienst van Anthropic is nu druk. Probeer het zo opnieuw."),
    (r"timed out|timeout|connection error", "The connection to Anthropic failed or timed out.",
     "De verbinding met Anthropic mislukte of duurde te lang."),
]


def error_hint(error: str, english: bool = False) -> str | None:
    """The likely cause of an API error in plain words, or None when it doesn't look familiar.
    Matched on the API's own error codes and types, not on any digits (a request id can hold "401")."""
    for pattern, en, nl in ERROR_HINTS:
        if re.search(pattern, error, re.I):
            return en if english else tr(en, nl)
    return None


def _private_copy(result):
    """A run on a visitor's own file, as it may be logged: timings, tokens, cost, tool names and
    the kind of error, but no question, answer, tool output or error text (all can hold their data)."""
    return dataclasses.replace(
        result, question="[own data: not logged]", answer="[own data: not logged]",
        tool_calls=[ToolCall(c.name, {}, {}, c.seconds) for c in result.tool_calls],
        error=None if result.error is None else (error_hint(result.error, english=True) or "unrecognised error"))


def _report_error(result, private: bool = False) -> None:
    """Write a failed answer's real reason to the app log (Streamlit Cloud: Manage app → logs)."""
    if result.stop_reason == "error" and result.error:
        detail = _private_copy(result).error if private else result.error
        print(f"[assistant] error after {result.seconds:.1f}s on {result.model}: {detail}",
              file=sys.stderr, flush=True)


def _show_details(result) -> None:
    """The 'How I got this' box: lets a manager (or reviewer) check where numbers came from."""
    with st.expander(tr("How I got this", "Hoe ik hieraan kom"), expanded=result.stop_reason == "error"):
        if result.stop_reason == "error" and result.error:
            hint = error_hint(result.error)
            if hint:
                st.markdown(tr(f"**Likely cause:** {hint}", f"**Waarschijnlijke oorzaak:** {hint}"))
            st.code(result.error[:400], language=None, wrap_lines=True)
        if result.tool_calls:
            for call in result.tool_calls:
                st.markdown(f"- `{call.name}` {json.dumps(call.input)} ({call.seconds:.2f} s)")
        else:
            st.markdown(tr("No tools were used.", "Er zijn geen hulpmiddelen gebruikt."))
        cost = cost_usd(result.model, result.input_tokens, result.output_tokens)
        cost_text = f"{cost:.5f} USD" if cost is not None else tr("unknown", "onbekend")
        st.caption(tr(f"{result.seconds:.1f} s total · {result.input_tokens} in / "
                      f"{result.output_tokens} out tokens · cost {cost_text} · {result.model}",
                      f"{result.seconds:.1f} s totaal · {result.input_tokens} in / "
                      f"{result.output_tokens} uit tokens · kosten {cost_text} · {result.model}"))


def _show_turn(question: str, result) -> None:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(_reply_text(result))
        _show_details(result)


def _asked_this_session() -> int:
    """Questions asked in this session over every conversation (sample hotel and own files)."""
    return sum(len(v) for k, v in st.session_state.items()
               if isinstance(k, str) and k.startswith(HISTORY_KEY) and isinstance(v, list))


def render(backend, history_key: str = HISTORY_KEY, private: bool = False) -> None:
    """Draw the chat page. Safe without an API key: it explains and disables input.

    history_key keeps one conversation per dataset, so answers about the sample hotel never
    become context for questions about the visitor's own file. The session cap counts them all.
    private: the backend holds the visitor's own data, so the run log keeps no content.
    """
    st.subheader(tr("Ask the assistant", "Vraag het de assistent"))
    st.caption(tr("Questions about expected guests and staffing, in English or Dutch.",
                  "Vragen over verwachte gasten en personeel, in het Nederlands of Engels."))

    history: list[tuple[str, object]] = st.session_state.setdefault(history_key, [])
    api_key = _api_key()
    counter = _daily_counter()
    session_limit = _asked_this_session() >= MAX_QUESTIONS_PER_SESSION
    daily_limit = counter.reached(MAX_QUESTIONS_PER_DAY)

    if not api_key:
        st.info(tr("The assistant needs an Anthropic API key. Add ANTHROPIC_API_KEY to "
                   ".streamlit/secrets.toml or set it as an environment variable.",
                   "De assistent heeft een Anthropic API key nodig. Zet ANTHROPIC_API_KEY in "
                   ".streamlit/secrets.toml of als omgevingsvariabele."))
    if session_limit:
        st.info(tr(f"You've asked {MAX_QUESTIONS_PER_SESSION} questions, the limit for one "
                   "session of this demo. Thanks for trying it!",
                   f"Je hebt {MAX_QUESTIONS_PER_SESSION} vragen gesteld, het maximum voor één sessie "
                   "van deze demo. Bedankt voor het uitproberen!"))
    elif daily_limit:
        st.info(tr(DAILY_LIMIT_NOTE, DAILY_LIMIT_NOTE_NL))
    disabled = not api_key or session_limit or daily_limit

    question = None
    with st.container(horizontal=True):
        for example in tr(EXAMPLE_QUESTIONS, EXAMPLE_QUESTIONS_NL):
            if st.button(example, disabled=disabled):
                question = example

    for past_question, past_result in history:
        _show_turn(past_question, past_result)

    typed = st.chat_input(tr("Ask about guests or staffing...", "Vraag naar gasten of personeel..."),
                          disabled=disabled, key="assistant_input")
    question = question or typed
    if not question or disabled:
        return

    if not counter.take(MAX_QUESTIONS_PER_DAY):  # another session took the last one
        st.info(tr(DAILY_LIMIT_NOTE, DAILY_LIMIT_NOTE_NL))
        return

    # Earlier turns give follow-ups like "and Sunday?" their context; the loop keeps the last 4.
    earlier = [{"question": q, "answer": r.answer} for q, r in history if r.stop_reason != "error"]
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        reply = st.empty()  # filled piece by piece while the answer streams in
        pieces: list[str] = []

        def show_piece(piece: str | None) -> None:
            if piece is None:  # that round called tools: its text was a preamble, not the answer
                pieces.clear()
                reply.empty()
                return
            pieces.append(piece)
            reply.markdown("".join(pieces) + " ▌")

        with st.spinner(tr("Checking the forecast...", "De voorspelling wordt bekeken...")):
            result = answer(question, backend, client=_client(api_key), history=earlier,
                            on_text=show_piece)
        log_run(_private_copy(result) if private else result)
        _report_error(result, private)
        reply.markdown(_reply_text(result))  # the final text (or a plain error), without the cursor
        _show_details(result)
    history.append((question, result))
    # This answer may have used the last question. The input above was drawn before it, still
    # enabled, so draw the page again: it then shows the limit note and disables the input.
    if _asked_this_session() >= MAX_QUESTIONS_PER_SESSION or counter.reached(MAX_QUESTIONS_PER_DAY):
        st.rerun()
