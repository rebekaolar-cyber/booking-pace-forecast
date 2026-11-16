"""The whole Streamlit app runs: every tab renders without an exception.

This replaces a health-check ping, which answers "ok" before the script has run at all.
"""
import pytest

from pathlib import Path

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_the_app_runs_with_all_three_tabs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=300).run()

    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert any("Plan" in label for label in labels)
    assert any("accurate" in label for label in labels)
    assert any("assistant" in label for label in labels)
    # Without a key the assistant explains itself and disables input instead of crashing.
    assert any("API key" in i.value for i in at.info)
    assert all(c.disabled for c in at.chat_input)
