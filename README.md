# rag1 — a tiny local RAG playground

A minimal Retrieval-Augmented Generation system that runs **100% on your machine** —
no API keys, no cloud, no cost. It indexes your own documents and answers questions
about them, citing the passages it used.

- **Embeddings + generation:** [Ollama](https://ollama.com) (`nomic-embed-text` + `llama3.2`)
- **Vector store:** numpy cosine similarity (no database)
- **Inputs:** `.txt`, `.md`, `.pdf` — point it at any file or folder (recursive)
- **Interfaces:** a streaming web UI *and* a CLI
- **Dependencies:** just `numpy` + `pypdf`; the web server is pure Python stdlib (no Flask/FastAPI)

## How it works

```
your files / folders   (data/, or any path you paste — recursive)
      │  loader.py        read .txt / .md / .pdf
      ▼
   chunker.py             context-anchored chunks (each prefixed with its doc's identity)
      │
      ▼  ollama embed     "search_document: " + chunk  → nomic-embed-text → vectors
   store.py               save index/  (embeddings.npy + records.json)
      │
ask "question?"
      │  ollama embed     "search_query: " + question → vector
      ▼  pipeline.retrieve dense cosine ⊕ BM25 lexical, fused (RRF)  → top-k chunks
   ollama chat            llama3.2 answers using only those chunks  → answer + sources
```

The two retrieval ideas that make this work well on real documents:

- **Asymmetric embedding prefixes.** `nomic-embed-text` is trained to receive a task
  instruction; documents and queries use *different* prefixes (`search_document:` /
  `search_query:`) so they align. Omitting them runs the model out-of-distribution.
- **Hybrid retrieval.** Dense vectors capture meaning; **BM25** captures exact tokens —
  company names, IDs, version strings — where embeddings are weakest. The two rankings are
  fused with Reciprocal Rank Fusion, so a hit strong in either retriever surfaces.

## Project layout

```
rag/
  config.py         models, paths, chunk/retrieval knobs (all env-overridable)
  loader.py         read .txt/.md/.pdf from any file or folder (recursive)
  chunker.py        split text into overlapping, context-anchored chunks
  ollama_client.py  Ollama HTTP calls: embed(prefix=…), chat_stream(), health()
  store.py          numpy vector store: cosine scores/search, save/load, upsert
  lexical.py        BM25 lexical retriever + Reciprocal Rank Fusion (pure numpy)
  pipeline.py       shared retrieval (dense ⊕ BM25) + prompt (used by ask.py and serve.py)
  indexer.py        ingest_paths(): load → chunk → embed → upsert → save
ingest.py           CLI: build/extend the index
ask.py              CLI: query the index (one-shot or REPL)
analyze.py          CLI: profile a doc set + sweep chunk sizes → recommend one (no embedding)
eval.py             CLI: measure retrieval quality (recall@k / MRR) against a question set
serve.py            web server (stdlib only): /api/status, /api/ingest, /api/ask
web/index.html      single-file UI (no build step, no CDN)
eval/questions.json question set for eval.py ({q, expect} pairs)
data/               default folder to drop documents in
index/              generated vector index (embeddings.npy + records.json)
```

## Setup (one time)

**1. Install Ollama** and start it:

```bash
brew install ollama       # macOS
ollama serve              # leave running in a terminal (or it runs as a background service)
```

**2. Pull the two models** (~2.5 GB total):

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

**3. Python deps** (a virtualenv `.venv` is already created here):

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Use it — web UI (easiest)

```bash
.venv/bin/python serve.py          # → http://127.0.0.1:8000
```

Open the page, **paste a file or folder path** (searched recursively), click **Ingest**,
then ask questions. It binds to `127.0.0.1` only. Features:

- **Streaming answers** with clickable `[1]`/`[2]` citation chips that jump to the source.
- **Source snippets** — every answer lists the retrieved chunks with similarity scores; expand to read them.
- **Backend indicator** (top-right pill): 🟢 connected · 🟡 model missing · 🔴 Ollama offline · ⚪ server offline.
  When connected it shows the Ollama version and the LLM build, e.g. `Ollama 0.30.8 · LLM llama3.2 (3.2B, Q4_K_M)` (hover the model for an explanation).
- **Collapsible ingest panel** — hidden by default; the `＋ Ingest` chip opens it. Includes a
  **Preview** (dry-run: counts files/chunks without embedding), per-ingest **chunk size/overlap**,
  and the **`replace`** toggle (*off*: add/update this path; *on*: wipe and rebuild from only it).
- **Collection questions** — "how many docs?", "what companies?" are answered from a computed
  index overview, since semantic search alone can't count or aggregate.

### Logs

Ingest progress and search results are visible in the UI, and the server also prints `[rag]`
activity lines to its stdout — each search logs the question and the retrieved chunks with scores:

```
  [rag] ingest done: 3 file(s), 12 new chunk(s), total 12
  [rag] ask: "what are the main points?" -> notes.md#4(0.81), notes.md#2(0.77)
```

Run `serve.py` in a terminal to watch them live.

## Use it — command line

```bash
# Ingest a folder (recursive) or a single file you point at:
.venv/bin/python ingest.py ~/Documents/notes
.venv/bin/python ingest.py ~/papers/paper.pdf ~/wiki

# ...or just drop files into data/ and run with no arguments:
.venv/bin/python ingest.py

# Ask away:
.venv/bin/python ask.py "what are the main points?"
.venv/bin/python ask.py            # interactive REPL
```

**Re-ingesting a path updates it** (no duplicate chunks); ingesting a new path
**adds** to the index. Use `--replace` to wipe the index and rebuild from scratch:

```bash
.venv/bin/python ingest.py --replace ~/papers
```

Extra options: `--dry-run` (scan + count without embedding or saving) and `--chunk-size N` /
`--chunk-overlap N` (override chunking for one run):

```bash
.venv/bin/python ingest.py --dry-run ~/papers                  # preview: file & chunk counts
.venv/bin/python ingest.py --chunk-size 1200 --replace ~/papers
```

## Pick good settings (measure, don't guess)

Two helper CLIs turn chunk-size and retrieval choices into numbers, before and after you embed:

```bash
# 1) Profile a doc set and get a chunk-size recommendation (no embedding, no Ollama):
.venv/bin/python analyze.py ~/papers
#    → length distribution + a chunk-size sweep (chunks/doc, % docs in ≤3 chunks, fill)

# 2) Re-ingest at the suggested size, then score retrieval against a question set:
.venv/bin/python ingest.py --replace --chunk-size 3200 --chunk-overlap 480 ~/papers
.venv/bin/python eval.py                       # recall@k + MRR over eval/questions.json
RAG_HYBRID=0 .venv/bin/python eval.py          # A/B: dense only vs dense+BM25 (default)
```

`eval/questions.json` is a list of `{"q": "...", "expect": "<substring of the source label>"}`;
a question "hits" when any retrieved passage's source contains `expect`. The defaults below
were chosen this way on the bundled job-post corpus (chunk size 3200 won a measured sweep).

## Tuning

All knobs are env vars (see `rag/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_GEN_MODEL` | `llama3.2` | generation model (try `llama3.1:8b`, `qwen2.5`, `mistral`) |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `RAG_EMBED_DOC_PREFIX` | `search_document: ` | task prefix added to documents at index time (set `""` for a model that doesn't use prefixes) |
| `RAG_EMBED_QUERY_PREFIX` | `search_query: ` | task prefix added to the question at ask time |
| `RAG_TEMPERATURE` | `0.2` | generation sampling temp — low = consistent, grounded answers (set `0` for fully deterministic) |
| `RAG_TOP_K` | `5` | chunks returned per question |
| `RAG_HYBRID` | `1` | blend BM25 lexical ranking with dense (`0` = dense only) |
| `RAG_RETRIEVE_POOL` | `20` | candidates each retriever contributes before fusion |
| `RAG_LEXICAL_MAX_DF_RATIO` | `0.5` | a query term only gets a BM25 vote if it appears in ≤ this fraction of chunks (filters common-word noise) |
| `RAG_MIN_SCORE` | `0.6` | off-topic gate: if even the *best* hit is below this cosine score, fall back to the corpus overview (`0` = never) |
| `RAG_MIN_SCORE_RATIO` | `0.5` | tail floor: drop returned hits below this fraction of the top hit's score |
| `RAG_CHUNK_SIZE` | `3200` | characters per chunk (tuned for short docs; use `analyze.py` for long-form) |
| `RAG_CHUNK_OVERLAP` | `480` | overlap between chunks |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Changing the embedding model, **the embedding prefixes**, or chunk settings? Re-run
`ingest.py --replace` to rebuild the index (it records the embed model + doc prefix and
refuses to mix incompatible ones).
