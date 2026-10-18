"""Retrieval on the real handbook: does the right page come back, and do off-topic
questions get "not covered"? No API key needed."""

import pytest

from handbook_qa import config
from handbook_qa.qa import ask

# Question -> handbook pages that hold the answer (checked by hand).
ON_TOPIC = [
    ("How much parental leave do I get?", {982, 983}),
    ("Does PostHog match 401k contributions?", {900}),
    ("What pension does PostHog offer in the UK?", {901}),
    ("Which laptop should an engineer get?", {971}),
    ("How much can I spend per month without asking?", {967}),
    ("Is there a limit on how many vacation days I can take?", {980, 981}),
]

NOT_COVERED = [
    "What's the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2022 World Cup?",
    "What's the weather in London tomorrow?",
    "What is the office dress code?",
    "Can I bring my dog to work?",
]


@pytest.mark.parametrize(("question", "pages"), ON_TOPIC)
def test_answer_page_is_in_top_k(handbook_index, question, pages):
    hits = handbook_index.search(question, config.TOP_K)
    retrieved = {p for hit in hits for p in hit.chunk.pages}
    assert retrieved & pages, f"expected one of {sorted(pages)}, got {sorted(retrieved)}"


@pytest.mark.parametrize(("question", "pages"), ON_TOPIC)
def test_on_topic_question_passes_the_cutoff(handbook_index, question, pages):
    answer = ask(question, handbook_index, client=None)
    assert answer.covered
    assert answer.mode == "retrieval_only"
    assert answer.citations


@pytest.mark.parametrize("question", NOT_COVERED)
def test_off_topic_question_is_not_covered(handbook_index, question):
    answer = ask(question, handbook_index, client=None)
    assert not answer.covered
    assert answer.citations == ()
