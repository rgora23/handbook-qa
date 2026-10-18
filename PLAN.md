# Handbook Q&A — v1 plan

A terminal app that answers questions about the PostHog handbook. Answers come only from the handbook and cite page numbers. If the handbook doesn't cover a question, the app says so.

Planned 2026-10-17. Due Sunday 2026-10-18. Budget: about 3 hours.

## The source document

- `Sample handbook.pdf` is the PostHog handbook, printed from Chrome.
- 1,076 pages, about 2M characters. Every page has extractable text, so no OCR is needed.
- Each section starts with a title line and a source path, like `contents/handbook/brand/art-requests.md`.
- Some people's names are missing from the extracted text (page 1 reads "handled by , , and").

## Decisions

| Area | Decision | Why |
|---|---|---|
| Language | Python 3.14. Fall back to 3.12 if a library has no 3.14 build. | Shortest path, easy to explain. |
| PDF text | `pypdf`, extracting one page at a time. | Per-page extraction is how every chunk knows its page. |
| Answer model | Claude, `claude-sonnet-5-5` by default, overridable by env var. | Good at "answer only from these excerpts." |
| API key | Read from `ANTHROPIC_API_KEY`. Not provided yet. | Reviewers use their own key. |
| Missing key | Print one clear line, then run in retrieval-only mode: show the top passages with pages. | The app stays useful, and retrieval can be tested without a key. |
| Embeddings | Local model via `fastembed` (bge-small, 384 numbers per chunk). | No second key, works offline, free to embed 2M characters. |
| Vector search | Plain `numpy`: cosine similarity against every chunk. | About 1,000 chunks × 384 numbers, roughly 1.5MB. Brute force is instant. No vector DB needed. |
| Chunk size | Medium: 2,000 characters (about 420 tokens), 300 overlap (~15%). Started at 2,400; changed after testing, see "Decisions from testing". | Middle ground between sharp citations and keeping a policy whole. Counted in characters so no tokenizer is needed. |
| Chunk metadata | Each chunk stores its page(s), its section title (from the `.md` path line) and its source file. | Pages power citations. Section titles make sources readable. The source file field makes multi-doc cheap later. |
| Chunks retrieved | `top_k = 5` (about 3k tokens of context). | The usual default. Enough recall without burying the answer. |
| "Not covered" rule | A score cutoff. If the best chunk scores below it, reply "the handbook doesn't cover this" and don't call Claude. | Cheap and testable without a key. We'll see how it performs. |
| Cutoff value | `MIN_SCORE = 0.60`, chosen in hour 1 from real scores. See "Decisions from testing". | bge scores bunch together (often 0.5–0.8), so the number has to come from data. |
| Prompt backstop | The prompt also lets Claude say the retrieved chunks don't answer the question. | Covers on-topic chunks that don't hold the answer. Not a second rule. |
| Citations | Inline `[p. N]` after each claim. Then a Sources list: page, section title, one-line quote. | Readers can check a claim without opening the PDF. Pages are PDF page numbers, what any viewer shows. |
| Interface | Terminal REPL, plus one-shot `ask "..."`. | Fits the time budget. A web UI adds nothing to grounding. |
| Core shape | One function, `ask(question) -> Answer`, in a module the CLI imports (e.g. `handbook_qa/qa.py`). It returns data and never prints. `Answer` holds `text` (with `[p. N]` cites), `covered`, `mode` (`answer` or `retrieval_only`) and `citations` (page, section, quote, score). Claude outages and rate limits raise typed errors. | The CLI only formats an `Answer`. An HTTP API later is a thin wrapper over the same function. |
| `--pdf` flag | Defaults to the handbook in the repo. Lets you point at another PDF. | Lets tests use a tiny fixture PDF instead of indexing 1,076 pages. |
| Memory | None. Each question stands alone. | Follow-ups need the question rewritten before search, which is real work. |
| Index cache | Saved to disk, keyed by the PDF's hash plus chunk settings. | Restarts take seconds, not minutes. |
| PDF in repo | Commit it (49.6MB, under GitHub's 50MB warning). | A clone runs with no download step. |
| Tests | pytest with 5–6 questions and their expected pages, plus one off-topic question that must come back "not covered". Retrieval-level only. Chunker tests use a small fixture PDF. | Nothing needs a key. Answer quality gets checked by hand once there's a key. |

## Decisions from testing

Changes made during the build because of what we measured, not planned up front.

| Area | Decision | What we tested | Why |
|---|---|---|---|
| Print footer | Strip Chrome's print footer from every page in the loader (`pdf_loader._clean`). | All 1,076 pages end with the same footer, e.g. `4/20/26, 4:49 PM PostHog handbook` / `file:///tmp/handbook-pdf-build/handbook.html 982/1076`. It's 87 characters, or 35 bge tokens. Its printed page number matched our PDF page number on all 1,076 pages. | It carries no content but costs 35–70 tokens per chunk (one or two footers), 7–14% of the 512 tokens the search model reads. The page-number match confirms our citations line up with the printed numbers, so we lose nothing by removing it. |
| Cutoff value | `MIN_SCORE = 0.60` in `config.py`. | Best-chunk scores for 13 questions. Off-topic (capital of France, sourdough, World Cup, London weather): 0.455–0.541. On-topic (parental leave, 401k, UK pension, laptop, spending limit, vacation days): 0.655–0.778. Near-miss: dress code 0.576, dog at work 0.579, company car 0.726. | 0.60 sits in the middle of the gap between off-topic and on-topic, and also catches 2 of the 3 near-misses. "Company car" scores above two real questions, so no cutoff catches it. The prompt backstop has to. |
| Chunk size | 2,000 characters with 300 overlap, down from 2,400 / 360. | bge-small reads only the first 512 tokens of a chunk. At 2,400: 378 of 952 chunks ran over, and 29 lost more than the overlap repeats, so part of their text couldn't be found by search. At 2,000: 1,146 chunks, median 419 tokens, 14 over, and only 2 lose more than the overlap. | Almost every word is now searchable. The cost is a bit less text in the top 5 (about 10,000 characters instead of 12,000). |
| Hybrid search | Add BM25 keyword search next to vectors. Merge the two rankings with reciprocal rank fusion (each ranking votes 1 / (60 + rank) for its top 50). The cutoff stays on the vector score only. | Vector-only, the 401k question put page 900 at rank 17 (rank 8 after the chunk-size change). Embeddings don't treat "401k" as special. | With hybrid search, all 6 test questions land in the top 5. 401k moved to rank 1. "Spend without asking" dropped from rank 1 to rank 5, still in. BM25 scores aren't on a 0–1 scale, so the cutoff stays on vector scores. |
| Cutoff re-check | Keep `MIN_SCORE = 0.60` after the two changes above. | Reran the 13 questions. Off-topic: 0.478–0.521. On-topic: 0.657–0.772. Near-miss: dress code 0.536, dog 0.561, company car 0.754. | 0.60 still sits in the gap and still catches 2 of 3 near-misses. |

## How it works

1. **At startup:** read the PDF page by page, chunk it, embed each chunk, and store the result as a matrix. Load it from the cache if the PDF and settings haven't changed.
2. **Per question:** rank every chunk two ways, by meaning (vector score) and by exact words (BM25). Merge the two rankings with reciprocal rank fusion and keep the top 5.
3. **If no kept chunk has a vector score at or above the cutoff:** say the handbook doesn't cover it.
4. **Otherwise:** send the 5 chunks to Claude with instructions to answer only from them and cite pages. With no key, show the chunks themselves.

## Out of scope for v1

- **Multiple source documents.** Stretch goal. The cheap part is about 30 minutes: chunks already carry a `source` field, citations would read `file.pdf p. N`, and `--pdf` would take several files. The open product questions are what to do when documents disagree, whether one cutoff works across documents, and how a user scopes a question to one document.
- **HTTP API.** Stretch goal, only if hour 3 finishes early. FastAPI app run with `uvicorn handbook_qa.api:app`. `POST /ask` takes `{"question": "..."}` and returns the `Answer` as JSON. The index loads once at startup. `GET /health` reports when it's ready. Empty question → 422, Claude outage → 502, rate limit → 429, no key → 200 in retrieval-only mode. The route is a plain `def` so blocking calls run in FastAPI's thread pool. One `TestClient` test. Plain JSON first, streaming later.
- **Follow-up questions with memory.** First next step after v1.
- **A web UI.**

## Known limitations to put in the README

- Some names are missing from the extracted text, so "who owns X" questions are weak.
- No follow-up memory.
- Search matches meaning and exact words, but not paraphrases that share neither ("spend without asking" vs "User Limit of $5,000 per month").
- Single document only.
- Answer quality is unverified until there's an API key.

## Order of work

1. **Hour 1: end to end.** Load, chunk, embed, search, answer path, citations. Without a key, the answer code is written but unverified.
2. **Hour 2: polish.** Error handling, the missing-key path, and setting the cutoff.
3. **Hour 3: finish.** Tests and README.
