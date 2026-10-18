"""The answer path, with a fake Claude. Uses the 3-page Time off fixture (pages 1-3)."""

import anthropic
import httpx
import pytest

from handbook_qa import llm
from handbook_qa.qa import NOT_COVERED_TEXT, ask
from tests.conftest import FakeClaude

QUESTION = "How much parental leave do I get?"
_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_answer_cites_pages_it_was_shown(time_off_index):
    claude = FakeClaude("Up to 24 weeks after a year [p. 3]. You can book it in Deel [p. 2].")
    answer = ask(QUESTION, time_off_index, claude)

    assert answer.covered
    assert answer.mode == "answer"
    assert answer.text.startswith("Up to 24 weeks")
    assert [c.page for c in answer.citations] == [3, 2]
    assert all(c.section == "Time off" and c.quote for c in answer.citations)


def test_cite_to_a_page_claude_was_not_shown_is_dropped(time_off_index):
    claude = FakeClaude("Up to 24 weeks [p. 3]. Something else [p. 99].")
    answer = ask(QUESTION, time_off_index, claude)
    assert [c.page for c in answer.citations] == [3]


def test_prompt_has_excerpts_with_page_markers(time_off_index):
    claude = FakeClaude("Up to 24 weeks [p. 3].")
    ask(QUESTION, time_off_index, claude)

    [request] = claude.requests
    content = request["messages"][0]["content"]
    assert "<excerpt" in content
    assert "[p. 3]" in content
    assert content.endswith(f"Question: {QUESTION}")


def test_claude_saying_not_covered(time_off_index):
    answer = ask(QUESTION, time_off_index, FakeClaude("NOT_COVERED."))
    assert not answer.covered
    assert answer.text == NOT_COVERED_TEXT
    assert answer.citations == ()


def test_below_cutoff_never_calls_claude(time_off_index):
    claude = FakeClaude("should not be used")
    answer = ask("What's the capital of France?", time_off_index, claude)
    assert not answer.covered
    assert claude.requests == []


def test_retrieval_only_mode_without_a_client(time_off_index):
    answer = ask(QUESTION, time_off_index, client=None)

    assert answer.mode == "retrieval_only"
    assert answer.text == ""
    assert answer.citations
    keys = [(c.page, c.quote) for c in answer.citations]
    assert len(keys) == len(set(keys))  # no duplicate quotes


def test_empty_question_is_rejected(time_off_index):
    with pytest.raises(ValueError):
        ask("   ", time_off_index, client=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            anthropic.RateLimitError(
                "slow down",
                response=httpx.Response(429, request=_REQUEST, headers={"retry-after": "7"}),
                body=None,
            ),
            llm.RateLimited,
        ),
        (
            anthropic.AuthenticationError(
                "bad key", response=httpx.Response(401, request=_REQUEST), body=None
            ),
            llm.ClaudeConfigError,
        ),
        (
            anthropic.InternalServerError(
                "overloaded", response=httpx.Response(529, request=_REQUEST), body=None
            ),
            llm.ClaudeUnavailable,
        ),
        (anthropic.APIConnectionError(request=_REQUEST), llm.ClaudeUnavailable),
    ],
)
def test_api_errors_become_typed_errors(time_off_index, error, expected):
    with pytest.raises(expected):
        ask(QUESTION, time_off_index, FakeClaude(error=error))


def test_rate_limit_carries_retry_after(time_off_index):
    error = anthropic.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=_REQUEST, headers={"retry-after": "7"}),
        body=None,
    )
    with pytest.raises(llm.RateLimited) as caught:
        ask(QUESTION, time_off_index, FakeClaude(error=error))
    assert caught.value.retry_after == 7.0


def test_refusal_becomes_answer_error(time_off_index):
    with pytest.raises(llm.AnswerError):
        ask(QUESTION, time_off_index, FakeClaude("", stop_reason="refusal"))
