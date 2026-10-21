from handbook_qa.cli import _format
from handbook_qa.qa import Answer, Citation


def test_sources_show_section_when_there_is_one():
    answer = Answer(
        text="Up to 24 weeks [p. 982].",
        covered=True,
        mode="answer",
        citations=(Citation(page=982, section="Time off", quote="Parental leave is exceptional.", score=0.77),),
    )
    assert '  p. 982 · Time off: "Parental leave is exceptional."' in _format(answer)


def test_no_dangling_dot_when_pdf_has_no_sections():
    answer = Answer(
        text="",
        covered=True,
        mode="retrieval_only",
        citations=(Citation(page=1, section="", quote="Include a README.", score=0.59),),
    )
    out = _format(answer)
    assert "  p. 1 (score 0.59)" in out
    assert "·" not in out
