from handbook_qa.pdf_loader import _clean, load_pdf
from tests.conftest import TIME_OFF_PDF


def test_pages_are_numbered_from_one():
    doc = load_pdf(TIME_OFF_PDF)
    assert [p.number for p in doc.pages] == [1, 2, 3]
    assert doc.source == "time_off.pdf"


def test_real_pages_have_no_print_footer_or_ligatures():
    doc = load_pdf(TIME_OFF_PDF)
    for page in doc.pages:
        assert "file://" not in page.text
        assert "ﬀ" not in page.text  # the "ff" ligature
    assert doc.pages[0].text.startswith("Time off\n")


def test_clean_expands_ligatures():
    assert _clean("Time oﬀ and ﬁling") == "Time off and filing"


def test_clean_strips_chrome_print_footer():
    page = (
        "Last line of content.\n"
        "4/20/26, 4:49 PM PostHog handbook\n"
        "file:///tmp/handbook-pdf-build/handbook.html 982/1076"
    )
    assert _clean(page) == "Last line of content."


def test_clean_keeps_file_paths_in_body_text():
    text = "Open file:///tmp/notes.txt in your editor."
    assert _clean(text) == text
