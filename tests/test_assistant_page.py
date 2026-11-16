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
