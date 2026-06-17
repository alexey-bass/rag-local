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
- `rag/ollama_client.py` — `embed()`, `chat_stream()`, `health()`. Raises `OllamaError`
  with a user-friendly message. `health()` **never raises** (the UI polls it every 5s).
- `rag/loader.py` — `load_documents(paths)`: file or dir (recursive), `.txt/.md/.pdf`.
  Returns dicts `{path (absolute), source (display label), text}`.
- `rag/chunker.py` — paragraph-aware chunking with overlap.
- `rag/store.py` — `VectorStore`: L2-normalized embeddings + cosine search. `add()`,
  `search()`, `remove_paths()` (upsert), `save()`/`load()` → `index/`.
- `rag/pipeline.py` — `retrieve()`, `build_user_message()`, `SYSTEM_PROMPT`. **Shared** by
  `ask.py` and `serve.py`. Change retrieval/prompt logic HERE, not in a front-end.
- `rag/indexer.py` — `ingest_paths(paths, replace, emit)`: load → chunk → embed → upsert →
  save. **Shared** by `ingest.py` and `serve.py`. `emit(event)` streams progress.
- `ingest.py` / `ask.py` — CLIs. `serve.py` — stdlib web server. `web/index.html` —
  single-file UI (vanilla JS, no build, no CDN).

## Invariants (don't break these)

- **Upsert by file path**: a record carries the file's absolute `path`; re-ingesting a path
  calls `remove_paths()` then re-adds, so no duplicate chunks. `--replace` / "replace index"
  wipes the whole index instead.
- **`records` and `embeddings` rows are parallel** in `VectorStore`. Any op that drops
  records must reindex the matrix identically (see `remove_paths`).
- Embeddings are normalized at `add()` time → search is a plain dot product.
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
- **Changing the embedding model or chunk settings invalidates the index** — re-run
  `ingest.py`. The index records its `embed_model`; mixing models raises an error.

## Verifying (no formal test suite)

- Byte-compile: `.venv/bin/python -m py_compile serve.py rag/*.py`.
- Logic without Ollama: unit-check `loader`/`chunker`/`store` against a temp dir and fake
  vectors (loader paths, `remove_paths` upsert, cosine search round-trip).
- Endpoints: `curl` `/api/status`, and the streaming `/api/ingest` + `/api/ask`. Real
  embed/generate requires Ollama running (`curl localhost:11434/api/tags`).

## Notes

- Server binds to `127.0.0.1` only (local-only by design).
- `data/README.md` is itself a `.md`, so it gets indexed if you ingest `data/` — expected.
