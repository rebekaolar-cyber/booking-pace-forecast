"""The assistant chat page, run headless with Streamlit's AppTest and a canned answer (no API calls)."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import app_pages.assistant as page  # noqa: E402
from agent.contract import RunResult  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "app_pages" / "demo_assistant.py"


@pytest.fixture
def asked(monkeypatch):
    """Questions that reached the model. answer() is replaced, so nothing is sent anywhere."""
    questions = []

    def canned(question, backend, client=None, history=None, **kwargs):
        questions.append(question)
        return RunResult(question=question, answer=f"answer {len(questions)}",
                         model="claude-haiku-5-5", prompt_version="test", stop_reason="answered")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.setattr(page, "answer", canned)
    monkeypatch.setattr(page, "log_run", lambda result: None)
    page._daily_counter.clear()  # a fresh daily count for this test
    return questions


def test_input_is_disabled_after_the_last_allowed_question(asked):
    at = AppTest.from_file(str(DEMO), default_timeout=60).run()
    assert not at.chat_input[0].disabled
    for i in range(page.MAX_QUESTIONS_PER_SESSION):
        at.chat_input[0].set_value(f"question {i + 1}").run()
        assert not at.exception
    assert len(asked) == page.MAX_QUESTIONS_PER_SESSION
    # Straight after the 10th answer: the 10th answer is on screen, the input is off and the
    # limit is explained, so an 11th question can't be typed and silently dropped.
    assert at.chat_input[0].disabled
    assert all(button.disabled for button in at.button)
    assert any("the limit for one session" in info.value for info in at.info)
    assert any(md.value == f"answer {page.MAX_QUESTIONS_PER_SESSION}"
               for message in at.chat_message for md in message.markdown)


def test_an_api_error_shows_its_real_reason(monkeypatch):
    """A failed answer says what went wrong, on the page and in the app log, not only 'try again'."""
    error = "Error code: 400 - {'type': 'error', 'error': {'message': 'Your credit balance is too low'}}"

    def failing(question, backend, client=None, history=None, **kwargs):
        return RunResult(question=question, answer="Sorry, I couldn't reach the AI service just now. "
                         "Please try again in a moment.", model="claude-haiku-5-5", prompt_version="test",
                         stop_reason="error", error=error)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.setattr(page, "answer", failing)
    monkeypatch.setattr(page, "log_run", lambda result: None)
    page._daily_counter.clear()
    at = AppTest.from_file(str(DEMO), default_timeout=60).run()
    at.chat_input[0].set_value("How busy is Saturday?").run()
    assert not at.exception
    shown = " ".join(md.value for md in at.markdown)
    assert "run out of credits" in shown
    assert any("credit balance is too low" in c.value for c in at.code)


@pytest.mark.parametrize("error, cause", [
    ("Error code: 401 - invalid x-api-key", "rejected"),
    ("Error code: 529 - Overloaded", "busy"),
    ("Error code: 429 - rate_limit_error", "Too many"),
    ("something nobody has seen before", None),
    ("Error code: 500 - internal error, request_id req_01401abc", None),  # "401" inside an id
])
def test_error_hints_name_the_likely_cause(error, cause):
    hint = page.error_hint(error)
    assert (hint is None) if cause is None else (cause in hint)


def test_own_data_runs_are_logged_without_their_content():
    from agent.contract import ToolCall
    result = RunResult(question="How busy is Saturday at my place?", answer="Saturday: 212 guests.",
                       model="claude-haiku-5-5", prompt_version="v2", stop_reason="error",
                       tool_calls=[ToolCall("get_forecast", {"start_date": "2026-11-21"},
                                            {"days": [{"forecast_guests": 212}]}, 0.1)],
                       error="Error code: 400 - Your credit balance is too low (212 guests)")
    logged = page._private_copy(result)
    text = repr(logged)
    assert "212" not in text and "Saturday" not in text and "2026-11-21" not in text
    assert [c.name for c in logged.tool_calls] == ["get_forecast"]
    assert logged.error == "The Anthropic account behind this demo has run out of credits."
    assert logged.input_tokens == result.input_tokens and logged.seconds == result.seconds
