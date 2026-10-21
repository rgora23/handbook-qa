"""Terminal front end: one-shot `ask "question"` or an interactive session with `ask`."""

import argparse
import sys
from pathlib import Path

from handbook_qa import config, llm
from handbook_qa.index import Index, SetupError, is_cached, load_index
from handbook_qa.qa import Answer, Citation, ask

QUIT_WORDS = {"exit", "quit", "q"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ask", description="Ask questions about the company handbook."
    )
    parser.add_argument(
        "question", nargs="*", help="Ask one question and exit. Leave out for an interactive session."
    )
    parser.add_argument(
        "--pdf", type=Path, default=config.DEFAULT_PDF, help="Handbook PDF (default: the one in this repo)."
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        print(f"Can't find the handbook at {args.pdf}. Pass the path with --pdf.", file=sys.stderr)
        return 2

    client = llm.make_client()
    if client is None:
        print(
            "ANTHROPIC_API_KEY isn't set, so I'll show the most relevant handbook passages "
            "instead of writing answers.",
            file=sys.stderr,
        )
    if not is_cached(args.pdf):
        print(
            f"Indexing {args.pdf.name} for the first time. This takes a few minutes "
            "(and downloads a 64MB search model if needed).",
            file=sys.stderr,
        )
    try:
        index = load_index(args.pdf)
    except SetupError as e:
        print(e, file=sys.stderr)
        return 1

    if args.question:
        return _ask_and_print(" ".join(args.question), index, client)

    print("Ask a question about the handbook. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in QUIT_WORDS:
            return 0
        if question:
            _ask_and_print(question, index, client)


def _ask_and_print(question: str, index: Index, client) -> int:
    try:
        answer = ask(question, index, client)
    except llm.AnswerError as e:
        print(f"Sorry, I couldn't get an answer: {e}", file=sys.stderr)
        return 1
    print(_format(answer))
    return 0


def _format(answer: Answer) -> str:
    if not answer.covered:
        return answer.text
    if answer.mode == "retrieval_only":
        lines = ["Most relevant passages:"]
        for c in answer.citations:
            lines.append(f"\n  {_where(c)} (score {c.score:.2f})")
            lines.append(f'    "{c.quote}"')
        return "\n".join(lines)
    lines = [answer.text]
    if answer.citations:
        lines.append("\nSources:")
        lines += [f'  {_where(c)}: "{c.quote}"' for c in answer.citations]
    return "\n".join(lines)


def _where(c: Citation) -> str:
    # Section names come from this handbook's header lines; other PDFs have none.
    return f"p. {c.page} · {c.section}" if c.section else f"p. {c.page}"
