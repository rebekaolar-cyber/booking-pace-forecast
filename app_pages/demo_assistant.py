"""Try the assistant page on its own with fake data:

    streamlit run app_pages/demo_assistant.py
"""
import sys
from pathlib import Path

# `streamlit run` puts only this folder on the path; the repo root holds the `agent` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.fake_backend import FakeBackend  # noqa: E402
from app_pages.assistant import render  # noqa: E402

render(FakeBackend())
