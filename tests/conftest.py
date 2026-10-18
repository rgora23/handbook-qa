from pathlib import Path
from types import SimpleNamespace

import pytest

from handbook_qa import config
from handbook_qa.index import load_index

FIXTURES = Path(__file__).parent / "fixtures"
# Handbook pages 980-982 (Time off), saved as a 3-page PDF. Pages are numbered 1-3 here.
TIME_OFF_PDF = FIXTURES / "time_off.pdf"


@pytest.fixture(scope="session")
def handbook_index():
    """The real handbook index. Loads from .cache; the first run builds it (about 2 minutes)."""
    return load_index(config.DEFAULT_PDF)


@pytest.fixture(scope="session")
def time_off_index():
    return load_index(TIME_OFF_PDF)


class FakeClaude:
    """Stands in for anthropic.Anthropic. Records each request and returns a canned reply."""

    def __init__(self, reply: str = "", error: Exception | None = None, stop_reason: str = "end_turn"):
        self.reply, self.error, self.stop_reason = reply, error, stop_reason
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=self.reply)],
        )
