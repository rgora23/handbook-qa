"""The core: question in, Answer out. Returns data and never prints, so a CLI or an HTTP API
can sit on top of it."""

import re
from dataclasses import dataclass
from typing import Literal

import anthropic

from handbook_qa import config, llm
from handbook_qa.chunker import SECTION_PATH
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
        # Overlapping chunks can yield the same quote twice. Keep the first.
        citations = {}
        for hit in hits:
            c = _quote(question, hit)
            citations.setdefault((c.page, c.quote), c)
        return Answer(text="", covered=True, mode=mode, citations=tuple(citations.values()))

    reply = llm.generate(client, question, hits)
    if llm.is_not_covered(reply):
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


@dataclass(frozen=True)
class _Line:
    text: str
    page: int
    section: str

    @property
    def is_heading(self) -> bool:
        return len(self.text) < 40 and not self.text.endswith(_SENTENCE_ENDS + (":", ","))

    @property
    def ends_sentence(self) -> bool:
        return self.text.endswith(_SENTENCE_ENDS)


_SENTENCE_ENDS = (".", "!", "?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _quote(question: str, hit: Hit, page: int | None = None) -> Citation:
    """A sentence from the chunk (optionally from one page) that shares the most words with the question.

    PDF text wraps mid-sentence, so we pick the best line, then widen it to the whole sentence.
    Short heading lines ("Parental leave") only win if there's nothing else.
    """
    lines = _lines(hit, page)
    if not lines:
        lines = _lines(hit, None)
    terms = _terms(question)
    best = max(
        range(len(lines)),
        key=lambda i: (not lines[i].is_heading, len(terms & _terms(lines[i].text))),
    )

    # Widen to the sentence: back to the line after the previous sentence end, forward to the
    # next sentence end. Stay on one page and stop at headings.
    start = best
    while start > 0 and _joinable(lines[start - 1], lines[best]) and not lines[start - 1].ends_sentence:
        start -= 1
    end = best
    while end + 1 < len(lines) and not lines[end].ends_sentence and _joinable(lines[end + 1], lines[best]):
        end += 1

    # Lines can hold the end of one sentence and the start of the next. Keep the best sentence.
    joined = " ".join(line.text for line in lines[start : end + 1])
    quote = max(_SENTENCE_SPLIT.split(joined), key=lambda s: (len(s) >= 30, len(terms & _terms(s))))
    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[: MAX_QUOTE_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return Citation(page=lines[best].page, section=lines[best].section, quote=quote, score=hit.score)


def _lines(hit: Hit, page: int | None) -> list[_Line]:
    lines, offset = [], 0
    for raw in hit.chunk.text.splitlines(keepends=True):
        segment = hit.chunk.locate(offset)
        offset += len(raw)
        text = raw.strip()
        if SECTION_PATH.match(text):
            continue  # "contents/handbook/....md" header lines aren't worth quoting
        if text and (page is None or segment.page == page):
            lines.append(_Line(text=text, page=segment.page, section=segment.section))
    return lines


def _joinable(line: _Line, anchor: _Line) -> bool:
    return line.page == anchor.page and not line.is_heading


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _stem(word: str) -> str:
    """Crude suffix trim so "engineering" matches "engineer". Good enough for picking a quote."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word
