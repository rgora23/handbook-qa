"""The core: question in, Answer out. Returns data and never prints, so a CLI or an HTTP API
can sit on top of it."""

import re
from dataclasses import dataclass
from typing import Literal

import anthropic

from handbook_qa import config, llm
from handbook_qa.index import Hit, Index

NOT_COVERED_TEXT = "The handbook doesn't cover this."
MAX_QUOTE_CHARS = 200

_CITE = re.compile(r"\[p\.\s*(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "are", "how", "what", "does", "can", "our", "you", "your", "with",
    "that", "this", "from", "have", "has", "was", "were", "about", "when", "where", "who",
    "why", "which", "there", "their", "they", "get", "much", "many", "any", "into", "should",
}


@dataclass(frozen=True)
class Citation:
    page: int
    section: str
    quote: str  # one line from the handbook, to check a claim without opening the PDF
    score: float  # retrieval score of the chunk the quote came from


@dataclass(frozen=True)
class Answer:
    text: str  # with inline [p. N] cites; empty in retrieval-only mode
    covered: bool
    mode: Literal["answer", "retrieval_only"]
    citations: tuple[Citation, ...]


def ask(
    question: str,
    index: Index,
    client: anthropic.Anthropic | None,
    *,
    top_k: int = config.TOP_K,
    min_score: float | None = config.MIN_SCORE,
) -> Answer:
    """Answer from the handbook only. Raises llm.AnswerError if Claude is unreachable."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    mode = "answer" if client else "retrieval_only"

    hits = index.search(question, top_k)
    # Hits are in fused order, so the best vector score isn't always first.
    if not hits or (min_score is not None and max(h.score for h in hits) < min_score):
        return Answer(text=NOT_COVERED_TEXT, covered=False, mode=mode, citations=())

    if client is None:
        return Answer(
            text="",
            covered=True,
            mode=mode,
            citations=tuple(_quote(question, hit) for hit in hits),
        )

    reply = llm.generate(client, question, hits)
    if reply == llm.NOT_COVERED:
        return Answer(text=NOT_COVERED_TEXT, covered=False, mode=mode, citations=())
    return Answer(text=reply, covered=True, mode=mode, citations=_cited(reply, question, hits))


def _cited(reply: str, question: str, hits: list[Hit]) -> tuple[Citation, ...]:
    """One citation per page Claude cited, in the order it cited them."""
    citations = []
    for page in dict.fromkeys(int(p) for p in _CITE.findall(reply)):
        containing = [h for h in hits if page in h.chunk.pages]
        if not containing:
            continue  # Claude cited a page it wasn't shown. Drop it rather than show a bad source.
        best = max(containing, key=lambda h: h.score)
        citations.append(_quote(question, best, page=page))
    return tuple(citations)


def _quote(question: str, hit: Hit, page: int | None = None) -> Citation:
    """The line in the chunk (optionally on one page) that shares the most words with the question."""
    terms = _terms(question)
    best_overlap, best_line, best_segment = -1, "", hit.chunk.segments[0]
    offset = 0
    for line in hit.chunk.text.splitlines(keepends=True):
        segment = hit.chunk.locate(offset)
        offset += len(line)
        text = line.strip()
        if not text or (page is not None and segment.page != page):
            continue
        overlap = len(terms & _terms(text))
        if overlap > best_overlap:
            best_overlap, best_line, best_segment = overlap, text, segment
    if len(best_line) > MAX_QUOTE_CHARS:
        best_line = best_line[: MAX_QUOTE_CHARS - 1].rstrip() + "…"
    return Citation(
        page=best_segment.page, section=best_segment.section, quote=best_line, score=hit.score
    )


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}
