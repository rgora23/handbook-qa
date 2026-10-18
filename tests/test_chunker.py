import pytest

from handbook_qa.chunker import chunk_document
from handbook_qa.pdf_loader import Document, Page, load_pdf
from tests.conftest import TIME_OFF_PDF


def make_doc(*page_texts: str, title: str = "") -> Document:
    return Document(
        source="test.pdf",
        title=title,
        pages=[Page(number=i + 1, text=text) for i, text in enumerate(page_texts)],
    )


def words(start: int, count: int) -> str:
    """Filler text with distinct words, wrapped into lines like PDF text."""
    ws = [f"word{i}" for i in range(start, start + count)]
    return "\n".join(" ".join(ws[i : i + 10]) for i in range(0, len(ws), 10))


def test_chunks_stay_within_size_and_overlap():
    doc = make_doc(words(0, 400), words(400, 400))
    chunks = chunk_document(doc, size=500, overlap=100)

    assert len(chunks) > 3
    assert all(len(c.text) <= 500 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        # The start of each chunk repeats the end of the one before it.
        assert nxt.text[:40] in prev.text


def test_chunk_that_crosses_a_page_break_knows_both_pages():
    doc = make_doc("alpha " * 50, "beta " * 50)
    [chunk] = chunk_document(doc, size=2000, overlap=100)

    assert chunk.pages == (1, 2)
    assert chunk.locate(chunk.text.index("alpha")).page == 1
    assert chunk.locate(chunk.text.index("beta")).page == 2

    marked = chunk.text_with_page_markers()
    assert marked.startswith("[p. 1]\n")
    assert marked.index("[p. 2]") < marked.index("beta")


def test_section_titles_come_from_header_lines():
    doc = make_doc(
        "Benefits\ncontents/handbook/people/benefits.md\nWe match 401k contributions.",
        "More about benefits, on a page with no header.",
        "Why does PostHog exist? Our mission and\nstrategy\ncontents/handbook/why.md\nBecause.",
    )
    chunks = chunk_document(doc, size=60, overlap=10)

    by_page = {s.page: s.section for c in chunks for s in c.segments}
    assert by_page[1] == "Benefits"
    assert by_page[2] == "Benefits"  # carries over until the next header
    assert by_page[3] == "Why does PostHog exist? Our mission and strategy"  # wrapped title joined


def test_path_mentioned_in_body_text_is_not_a_header():
    doc = make_doc(
        "Docs\ncontents/handbook/docs.md\nline\nline\nline\n"
        "To edit it, open\ncontents/docs/api/annotations_list.mdx\nand save."
    )
    [chunk] = chunk_document(doc, size=2000, overlap=100)
    assert chunk.sections == ("Docs",)


def test_pdf_title_above_first_section_is_dropped():
    doc = make_doc(
        "PostHog handbook\nArt requests\ncontents/handbook/brand/art.md\nAsk the design team.",
        title="PostHog handbook",
    )
    [chunk] = chunk_document(doc, size=2000, overlap=100)
    assert chunk.sections == ("Art requests",)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(make_doc("text"), size=100, overlap=100)


def test_real_pdf_chunks_cover_every_page():
    chunks = chunk_document(load_pdf(TIME_OFF_PDF), size=2000, overlap=300)

    assert {p for c in chunks for p in c.pages} == {1, 2, 3}
    assert all(c.sections == ("Time off",) for c in chunks)
    assert all(c.source == "time_off.pdf" for c in chunks)
