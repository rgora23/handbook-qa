"""Split a document into overlapping, fixed-size chunks that remember their pages and sections.

The handbook is printed from markdown pages. Each section starts at the top of a PDF page:
a title (one or two lines) followed by its source path, e.g.

    Art, brand, and merch requests
    contents/handbook/brand/art-requests.md

Chunks don't stop at page or section boundaries. Instead each chunk keeps a list of
segments: where inside its text a new page or section begins. That lets us cite the exact
page a sentence came from, not just the chunk's first page.
"""

import bisect
import re
from dataclasses import dataclass

from handbook_qa.pdf_loader import Document

SECTION_PATH = re.compile(r"^contents/\S+\.mdx?\s*$", re.MULTILINE)
# A path line only counts as a section header if it's this close to the top of its page.
# Otherwise it's a file path mentioned in the body text.
MAX_HEADER_LINE = 3


@dataclass(frozen=True)
class Segment:
    offset: int  # where this page/section run starts inside Chunk.text
    page: int
    section: str


@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    source: str
    segments: tuple[Segment, ...]  # sorted by offset, first one at offset 0

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(s.page for s in self.segments))

    @property
    def sections(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.section for s in self.segments if s.section))

    def locate(self, offset: int) -> Segment:
        """The page and section that a character offset in this chunk belongs to."""
        i = bisect.bisect_right([s.offset for s in self.segments], offset) - 1
        return self.segments[max(i, 0)]

    def text_with_page_markers(self) -> str:
        """The chunk text with a [p. N] marker wherever a new page starts."""
        parts, last_page = [], None
        bounds = [s.offset for s in self.segments] + [len(self.text)]
        for seg, end in zip(self.segments, bounds[1:]):
            if seg.page != last_page:
                parts.append(f"[p. {seg.page}]\n")
                last_page = seg.page
            parts.append(self.text[seg.offset : end])
        return "".join(parts)


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError(f"overlap ({overlap}) must be between 0 and size ({size})")

    text, page_starts = _join_pages(doc)
    sections = _find_sections(doc, page_starts)
    # Every place the (page, section) pair can change, as (offset, page, section).
    boundaries = _boundaries(doc, page_starts, sections)
    boundary_offsets = [b[0] for b in boundaries]

    chunks: list[Chunk] = []
    for start, end in _windows(text, size, overlap):
        raw = text[start:end]
        body = raw.strip()
        if not body:
            continue
        body_start = start + (len(raw) - len(raw.lstrip()))
        body_end = body_start + len(body)
        chunks.append(
            Chunk(
                id=len(chunks),
                text=body,
                source=doc.source,
                segments=_segments(boundaries, boundary_offsets, body_start, body_end),
            )
        )
    return chunks


def _join_pages(doc: Document) -> tuple[str, list[int]]:
    """One string for the whole document, plus the offset where each page starts."""
    parts, page_starts, offset = [], [], 0
    for page in doc.pages:
        page_starts.append(offset)
        parts.append(page.text)
        offset += len(page.text) + 1  # +1 for the joining newline
    return "\n".join(parts), page_starts


def _find_sections(doc: Document, page_starts: list[int]) -> list[tuple[int, str]]:
    """(offset, title) for every section header."""
    sections = []
    for page, page_start in zip(doc.pages, page_starts):
        match = SECTION_PATH.search(page.text)
        if not match:
            continue
        title_lines = page.text[: match.start()].splitlines()
        if len(title_lines) > MAX_HEADER_LINE:
            continue
        # The first page also carries the PDF's own title above the section title.
        title_lines = [line.strip() for line in title_lines if line.strip() and line.strip() != doc.title]
        title = " ".join(title_lines) or match.group().strip()
        sections.append((page_start, title))
    return sections


def _boundaries(
    doc: Document, page_starts: list[int], sections: list[tuple[int, str]]
) -> list[tuple[int, int, str]]:
    section_offsets = [offset for offset, _ in sections]
    out = []
    for page, page_start in zip(doc.pages, page_starts):
        i = bisect.bisect_right(section_offsets, page_start) - 1
        out.append((page_start, page.number, sections[i][1] if i >= 0 else ""))
    return out


def _segments(
    boundaries: list[tuple[int, int, str]], offsets: list[int], start: int, end: int
) -> tuple[Segment, ...]:
    first = max(bisect.bisect_right(offsets, start) - 1, 0)
    last = bisect.bisect_right(offsets, end - 1) - 1
    return tuple(
        Segment(offset=max(offset - start, 0), page=page, section=section)
        for offset, page, section in boundaries[first : last + 1]
    )


def _windows(text: str, size: int, overlap: int):
    """Yield (start, end) spans of about `size` characters, ending on a line break or space."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Break at a newline in the last 20% of the window, else at a space.
            floor = start + int(size * 0.8)
            cut = text.rfind("\n", floor, end)
            if cut == -1:
                cut = text.rfind(" ", floor, end)
            if cut != -1:
                end = cut
        yield start, end
        if end >= len(text):
            return
        next_start = max(end - overlap, start + 1)
        # Don't start the next chunk mid-word.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
