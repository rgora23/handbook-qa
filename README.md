# Handbook Q&A

Ask questions about a company handbook in your terminal. Answers come only from the handbook and cite the pages they came from. If it can't find the answer in the handbook, it says so instead of guessing.

The source is the PostHog handbook PDF (1,076 pages), included as `Sample handbook.pdf`.

## Setup

Requires Python 3.12 or newer (developed on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To get written answers, set an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

Without a key the app still works. It shows the most relevant handbook passages, with page numbers, instead of a written answer.

## Running it

Ask one question:

```bash
ask "How much parental leave do I get?"
```

Or start an interactive session, and type `exit` or press Ctrl-D to quit:

```bash
ask
```

The first run indexes the handbook and downloads a 64MB search model. That takes 2–3 minutes. After that, startup takes about 2 seconds.

Options:

| Setting | Default | What it does |
|---|---|---|
| `--pdf PATH` | `Sample handbook.pdf` | Ask about a different PDF. |
| `ANTHROPIC_API_KEY` | not set | Turns on written answers from Claude. |
| `HANDBOOK_QA_MODEL` | `claude-sonnet-5-5` | Which Claude model writes answers. |
| `HANDBOOK_QA_EFFORT` | `low` | How hard Claude thinks before answering. |

### What you'll see

With a key, you get an answer with a `[p. N]` cite after each claim, then a Sources list:

```
Sources:
  p. 982 · Time off: "Parental leave is exceptional as it needs to be significantly longer than a typical vacation."
```

Without a key:

```
Most relevant passages:

  p. 982 · Time off (score 0.77)
    "Parental leave is exceptional as it needs to be significantly longer than a typical vacation."
  ...
```

Off-topic questions get `I couldn't find this in the handbook.`

## How it works

**Build, once, then cached in `.cache/`:**

1. **Load** (`pdf_loader.py`). Read the PDF one page at a time so text keeps its page number. Fix ligature characters ("oﬀ" → "off") and strip Chrome's print footer from every page.
2. **Chunk** (`chunker.py`). Cut the text into 2,000-character chunks with 300 characters of overlap: 1,146 chunks. Each chunk records where its page breaks and section changes fall, so a cite can name the exact page even when a chunk spans two pages.
3. **Index** (`index.py`). Embed each chunk with a small local model (bge-small, 384 numbers per chunk) and build a BM25 keyword index. The vectors are cached, keyed by the PDF's hash plus the chunk settings.

**Each question:**

4. **Search** (`index.py`). Rank every chunk two ways: by meaning (vector similarity) and by exact words (BM25). Merge the two rankings with reciprocal rank fusion and keep the top 5.
5. **Cutoff** (`qa.py`). If none of the 5 has a vector score of at least 0.60, answer "I couldn't find this in the handbook" without calling Claude.
6. **Answer** (`llm.py`). Send the 5 chunks to Claude with `[p. N]` markers inside them. The prompt says to answer only from them, cite a page after every claim, and reply `NOT_COVERED` if they don't hold the answer.
7. **Cite** (`qa.py`). Build Sources from the pages Claude cited, each with a one-sentence quote from that page. Cites to pages Claude wasn't shown are dropped from Sources.

`qa.ask()` returns an `Answer` object and never prints, so the CLI (`cli.py`) is a thin layer on top, and an HTTP API could be too.

[PLAN.md](PLAN.md) records every decision and the measurements behind it.

## Key decisions

- **A local embedding model, not an embeddings API.** Reviewers need one key, not two. Embedding about 2M characters is free and works offline after the first run.
- **Plain numpy, no vector database.** 1,146 × 384 numbers is about 1.7MB. Comparing a question against every chunk takes milliseconds.
- **2,000-character chunks.** bge-small only reads the first 512 tokens of a chunk. At 2,400 characters, 378 chunks ran over that limit, and 29 lost text the overlap didn't cover. At 2,000, only 14 run over, and 2 lose more than the overlap covers.
- **Hybrid search.** Vector search alone ranked the 401k page 17th for "Does PostHog match 401k contributions?", because embeddings don't treat exact terms as special. With BM25 added, it ranks 1st, and all 6 test questions find their page in the top 5.
- **A 0.60 cutoff on the vector score.** Measured on 13 questions. Off-topic questions scored 0.478–0.521 and real ones 0.657–0.772. BM25 scores have no fixed scale, so the cutoff ignores them.
- **Claude as a backstop.** Some questions pass the cutoff but aren't answered by the handbook ("Does PostHog give employees a company car?" scores 0.754). Claude replies `NOT_COVERED` for those.

## Tests

```bash
pytest
```

57 tests, none need an API key:
- **Retrieval on the real handbook:** 6 questions with hand-checked answer pages, and 6 off-topic or near-miss questions that must come back "not covered".
- **Chunker and loader:** made-up documents, plus a 3-page fixture cut from the handbook.
- **Answer path:** a fake Claude client. Covers cites, dropped cites, NOT_COVERED, the cutoff skipping Claude, and API errors mapped to typed errors.

The first run builds the handbook index (2–3 minutes) if it isn't cached yet.

## Assumptions

- **The PDF is the only source of truth.** Answers never use outside knowledge, even when it's correct.
- **Page numbers are the PDF's own.** They match the numbers printed in each page's footer.
- **The PDF has extractable text.** Scanned PDFs (images only) aren't supported. You get a clear message instead.
- **Each question stands alone,** with no memory of earlier questions.
- **"Not covered" beats a guess.** Saying the answer couldn't be found is better than a plausible guess.
- **One source document.** The brief asks about one company handbook, so v1 answers from a single PDF at a time. Supporting several is listed under Next steps.
- **One local user.** It's a terminal app for one person: no login, no multi-user access, no hosting.
- **English only.** The handbook and the questions are in English, so an English-only search model is enough.
- **The handbook is trusted.** It's the company's own document, so excerpts go to Claude as written, with no defence against instructions planted in the text.
- **Sending text to Anthropic is acceptable.** Each question and its 5 excerpts go to Claude's API. For an internal handbook, that's assumed fine.
- **First run needs internet.** Downloading the 64MB search model needs internet once. After that, search works offline. Written answers need your own API key.
- **Answers live in a few places.** The top 5 chunks are enough for most policy questions. A question that needs the whole handbook, like "summarise every benefit", gets a partial answer.

## Limitations

**What the app can't see in the PDF.** It only reads text extraction, so some content never reaches search or Claude:

- **Links lose their URLs.** The handbook has 2,546 links on 632 pages. Text extraction keeps a link's words ("request templates") but not the URL behind them, so answers can point to a resource but can't say where it lives. URLs printed as visible text (35 pages) do come through.
- **Text inside images is lost.** The handbook has 78 images on 62 pages. Screenshots and diagrams contribute nothing.
- **Tables are flattened.** Rows become lines and columns become spaces. The parental leave table on page 982 reads "Time at posthog maternity leave paternity leave / under 6 months 3 weeks 2 - 3 weeks". Claude can usually rebuild it, but it can misread which number belongs to which column.
- **Some names are missing.** The extracted text drops some people's names (page 1 reads "handled by , , and"), so "who owns X" questions are weak.
- **It's a snapshot.** The PDF was printed on April 20, 2026. Later handbook changes aren't reflected.

**Search:**

- **English only.** The search model is English-only. "¿Cuántas semanas de baja por paternidad tengo?" scores 0.564 and a German version 0.535, both under the cutoff. They get "not covered" even though the handbook answers them.
- **Paraphrases can rank low.** Search matches meaning and exact words. "How much can I spend per month without asking?" still finds its page, but only at rank 5.
- **Near-miss questions rely on Claude.** "Company car" scores above some real questions, so only the prompt backstop catches it.
- **The cutoff rests on a small sample.** 0.60 was chosen from 13 questions. Differently worded questions near the line could land on the wrong side.
- **Overlapping chunks crowd the top 5.** Neighbouring chunks share 300 characters, and in all 5 sample questions we checked, two of the top 5 covered the same page. That leaves less room for other relevant sections.

**Answers and citations:**

- **Citation checks are partial.** Cites to pages Claude wasn't shown are dropped from Sources but stay in the answer text. Quotes are picked by overlap with the question, not with the specific claim.
- **Sources name the top-level section only.** A cite shows "Time off", not the "Parental leave" subheading under it.
- **No memory.** A follow-up like "what about contractors?" searches on its own.

**Scope:**

- **One PDF at a time, and it's not built for more yet.** Each chunk records its source file and the cache is per file, but the rest assumes one document:
  - `--pdf` takes one path.
  - The index holds one document.
  - Neither the prompt nor the citations say which file a passage came from.
  - The cutoff was tuned on this handbook alone.
- **Tuned to this handbook's format.** Section names come from its `contents/....md` header lines, and the footer cleanup matches Chrome's print footer. Another PDF still runs with `--pdf`, but:
  - Sources show no section name.
  - The 0.60 cutoff doesn't transfer. On the take-home brief, "What should the README include?" scored 0.594 and got "not covered", even though the brief answers it.
- **PDF text is trusted.** Excerpts go to Claude as they are. A PDF with instructions written into it could try to steer the answer. That's fine for this handbook, but a risk with arbitrary files.

## Next steps

1. Follow-up questions: rewrite the question using the conversation before searching.
2. Grade real answers: a small eval set that checks the answer text and its cites, not only retrieval.
3. An HTTP API (`POST /ask`) on top of `qa.ask()`.
4. Several documents: index a list of PDFs, put the file name in each excerpt and cite, and set the cutoff per document.
5. Keep link URLs by matching each link's box to its words on the page.

## Project layout

```
handbook_qa/
  config.py       settings in one place
  pdf_loader.py   PDF -> cleaned pages
  chunker.py      pages -> chunks with page and section segments
  index.py        embeddings + BM25, cache, hybrid search
  llm.py          the only code that calls Claude
  qa.py           ask() -> Answer: cutoff, answer, citations
  cli.py          one-shot and interactive terminal front end
tests/            pytest suite and a 3-page fixture PDF
PLAN.md           decisions and the numbers behind them
```
