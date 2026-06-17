# CLAUDE.md

Guidance for working in this repo.

## What this is

**rag1** — a tiny, fully-local Retrieval-Augmented Generation playground. It indexes the
user's own documents and answers questions over them, citing sources. No cloud, no API
keys: embeddings *and* generation both run through a local **Ollama** server.

## Stack & dependencies

- Python 3.14, virtualenv at `.venv/`. Run things with `.venv/bin/python`.
- Third-party deps: **only** `numpy` + `pypdf` (see `requirements.txt`).
- `serve.py` is **pure Python stdlib** (`http.server`). The Ollama client uses `urllib`.
  **Do not add** Flask/FastAPI/requests/torch/sentence-transformers/chromadb — the design
  is deliberately minimal and Python-3.14-friendly.
- Backend: Ollama at `http://localhost:11434` (`ollama serve`), models `nomic-embed-text`
  (embeddings) and `llama3.2` (generation). Installed via Homebrew on this machine.

## Architecture — one code path, two front-ends

- `rag/config.py` — all settings; every knob is env-overridable (`RAG_*`, `OLLAMA_HOST`).
- `rag/ollama_client.py` — `embed(texts, prefix="")`, `chat_stream()`, `health()`. Raises
  `OllamaError` with a user-friendly message. `health()` **never raises** (UI polls it every 60s).
  `prefix` prepends nomic's task instruction (`search_document: ` / `search_query: `) to each
  input before sending — it never mutates the caller's stored text.
- `rag/loader.py` — `load_documents(paths)`: file or dir (recursive), `.txt/.md/.pdf`.
  Returns dicts `{path (absolute), source (display label), text}`.
- `rag/chunker.py` — **context-anchored, structure-aware** chunking: every chunk is prefixed
  with the document's identity (title + Company/Location/Track parsed from the front-matter)
  and split on line boundaries (no mid-word cuts). This is what makes entity/field queries work.
- `rag/store.py` — `VectorStore`: L2-normalized embeddings + cosine search. `scores()` (full
  similarity vector, used by hybrid), `search()` (top-k), `add()`, `remove_paths()` (upsert),
  `save()`/`load()` → `index/`. Persists `embed_model` **and** `embed_doc_prefix`.
- `rag/lexical.py` — `BM25` (Okapi, postings-based) + `rrf()` (Reciprocal Rank Fusion) +
  `tokenize()`. `BM25.discriminating()` drops corpus-wide-common query terms so lexical
  matching fires on distinguishing tokens. Pure stdlib + numpy.
- `rag/corpus.py` — `corpus_overview(store)`: computed collection stats (doc/chunk counts,
  top companies) used to answer aggregate/meta questions semantic search can't.
- `rag/pipeline.py` — `retrieve()`: embeds the question with `EMBED_QUERY_PREFIX`, gates on
  `MIN_SCORE` (best hit → else corpus fallback), fuses dense + BM25 via RRF when `HYBRID`,
  trims the tail by `MIN_SCORE_RATIO`. Also `build_user_message()`, `SYSTEM_PROMPT`,
  `OVERVIEW_PROMPT` + `overview_user_message()`. **Shared** by `ask.py` and `serve.py`.
  Change retrieval/prompt logic HERE, not in a front-end.
- `rag/indexer.py` — `ingest_paths(paths, replace, emit, chunk_size, chunk_overlap, dry_run)`:
  load → chunk → embed → upsert → save. `dry_run` scans + chunks + reports counts without
  embedding/saving; `chunk_size`/`chunk_overlap` override defaults per run. **Shared** by
  `ingest.py` and `serve.py`; `emit(event)` streams progress.
- `ingest.py` / `ask.py` — CLIs. `serve.py` — stdlib web server. `web/index.html` —
  single-file UI (vanilla JS, no build, no CDN).
- `analyze.py` — profiles a doc set (length distribution) and sweeps chunk sizes through the
  real chunker to **recommend** one; no embedding/Ollama needed. Run it before a big ingest.
- `eval.py` — measures retrieval (recall@k + MRR) for `retrieve()` against `eval/questions.json`
  (`{q, expect}` pairs; `expect` = substring of the source label). The way to A/B any
  chunk/embedding/hybrid change — e.g. `RAG_HYBRID=0 eval.py` vs default.

## Invariants (don't break these)

- **Upsert by file path**: a record carries the file's absolute `path`; re-ingesting a path
  calls `remove_paths()` then re-adds, so no duplicate chunks. `--replace` / "replace index"
  wipes the whole index instead.
- **`records` and `embeddings` rows are parallel** in `VectorStore`. Any op that drops
  records must reindex the matrix identically (see `remove_paths`).
- Embeddings are normalized at `add()` time → search is a plain dot product.
- **Asymmetric embedding prefixes**: documents are embedded with `EMBED_DOC_PREFIX`
  (`search_document: `), questions with `EMBED_QUERY_PREFIX` (`search_query: `). nomic-embed-text
  *requires* this; the two prefixes must stay paired. The index stamps `embed_doc_prefix`;
  `ingest_paths` refuses to append under a different one, and **changing either prefix invalidates
  the index → re-ingest with `--replace`** (same rule as changing the embed model).
- **Hybrid retrieval**: `retrieve()` fuses dense cosine with BM25 (RRF) when `config.HYBRID`.
  BM25 is built lazily from chunk text and cached on the store (`store._bm25`), rebuilt when the
  chunk count changes. The cosine `MIN_SCORE` gate and `MIN_SCORE_RATIO` tail floor still apply,
  so BM25 reorders/promotes within the topical set but can't smuggle in off-topic chunks.
- **Context anchoring**: `chunker.py` prefixes each chunk with its document's header so body
  chunks aren't anonymous. Re-chunking changes chunk text → **re-ingest** to take effect.
- **Score gate + tail floor**: `retrieve()` first gates on `config.MIN_SCORE` (default 0.6)
  applied to the **best** hit — if even that is below it, both front-ends fall back to
  `corpus_overview` (so "how many docs?" works; semantic search alone can't count/aggregate).
  Once past the gate it returns up to `TOP_K`, trimming hits below `MIN_SCORE_RATIO × top_score`.
- **Streaming protocol**: `/api/ask` and `/api/ingest` stream **NDJSON** (one JSON object
  per line) over an HTTP/1.0 connection-close response. Event `type`s: `status`, `loaded`,
  `chunked`, `embed`, `sources`, `token`, `done`, `error`. The browser reads it via
  `fetch().body.getReader()`; keep new events line-delimited.

## Running

```bash
.venv/bin/python serve.py            # web UI → http://127.0.0.1:8000
.venv/bin/python ingest.py PATH...   # build/extend index (no args = data/; --replace to wipe)
.venv/bin/python ask.py "question"   # one-shot, or no args for a REPL
```

## After making changes

- **`serve.py` / `rag/*.py` edits need a server restart**: `pkill -f "serve.py 8000"` then
  relaunch. `web/index.html` is re-read per request, so HTML/JS/CSS changes only need a
  browser refresh.
- **Changing the embedding model, the embedding prefixes, or chunk settings invalidates the
  index** — re-run `ingest.py --replace`. The index records its `embed_model` + `embed_doc_prefix`;
  appending under a mismatched model or prefix raises an error.
- **Keep docs in sync (project rule):** when you change app behavior, features, config knobs,
  or architecture, update `README.md` and this `CLAUDE.md` in the same change. Docs must
  reflect the current app.

## Verifying (no formal test suite)

- Byte-compile: `.venv/bin/python -m py_compile serve.py ask.py ingest.py eval.py analyze.py rag/*.py`.
- Logic without Ollama: unit-check `loader`/`chunker`/`store`/`lexical` against a temp dir and
  fake vectors (loader paths, `remove_paths` upsert, cosine round-trip, BM25 ranking, RRF). And
  `analyze.py PATH` runs the full chunker with no embedding — a quick sanity check on chunking.
- Retrieval quality: `eval.py` (needs the embed model) reports recall@k/MRR; the way to confirm
  a chunk/embedding/hybrid change actually helped before committing.
- Endpoints: `curl` `/api/status`, and the streaming `/api/ingest` + `/api/ask`. Real
  embed/generate requires Ollama running (`curl localhost:11434/api/tags`).

## Notes

- Server binds to `127.0.0.1` only (local-only by design).
- `data/README.md` is itself a `.md`, so it gets indexed if you ingest `data/` — expected.
