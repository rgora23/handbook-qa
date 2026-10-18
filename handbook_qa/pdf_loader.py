"""Read a PDF one page at a time, so every piece of text keeps its page number."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# pypdf logs warnings about malformed files on its own. The caller turns real failures into
# one clear message, so keep only errors.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, matching what a PDF viewer shows
    text: str


@dataclass(frozen=True)
class Document:
    source: str  # file name, shown in citations
    title: str  # from the PDF metadata, may be empty
    pages: list[Page]


def load_pdf(pdf_path: Path) -> Document:
    reader = PdfReader(pdf_path)
    title = (reader.metadata.title if reader.metadata else None) or ""
    pages = [
        Page(number=i + 1, text=_clean(page.extract_text() or ""))
        for i, page in enumerate(reader.pages)
    ]
    return Document(source=pdf_path.name, title=_clean(title).strip(), pages=pages)


# Chrome's print footer, at the end of every page:
#   4/20/26, 4:49 PM PostHog handbook
#   file:///tmp/handbook-pdf-build/handbook.html 982/1076
_PRINT_FOOTER = re.compile(
    r"\n?\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M[^\n]*\nfile://\S+ \d+/\d+\s*$"
)


def _clean(text: str) -> str:
    # The PDF uses ligature characters ("Time oﬀ" with a single "ﬀ"). NFKC turns them back
    # into plain letters so "off" in a question can match "off" in the text.
    text = unicodedata.normalize("NFKC", text)
    return _PRINT_FOOTER.sub("", text)
