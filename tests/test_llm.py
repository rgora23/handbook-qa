import pytest

from handbook_qa import llm


@pytest.mark.parametrize(
    "reply",
    ["NOT_COVERED", "NOT_COVERED.", "`NOT_COVERED`", "**NOT_COVERED**", "not_covered", "  NOT_COVERED \n", '"NOT_COVERED"'],
)
def test_not_covered_variants_are_recognized(reply):
    assert llm.is_not_covered(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Parental leave is 24 weeks [p. 982]. NOT_COVERED for contractors.",
        "The handbook says NOT_COVERED",
        "",
    ],
)
def test_real_answers_are_not_mistaken_for_not_covered(reply):
    assert not llm.is_not_covered(reply)


def test_no_key_means_no_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.make_client() is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert llm.make_client() is None


def test_key_means_a_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert llm.make_client() is not None
