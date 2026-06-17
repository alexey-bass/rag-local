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
      ▼  ollama embed     nomic-embed-text  → vectors
   store.py               save index/  (embeddings.npy + records.json)
      │
ask "question?"
      │  ollama embed     question → vector
      ▼  store.search     top-k most similar chunks  (cosine)
   ollama chat            llama3.2 answers using only those chunks  → answer + sources
```

## Project layout

```
rag/
  config.py         models, paths, chunk/retrieval knobs (all env-overridable)
  loader.py         read .txt/.md/.pdf from any file or folder (recursive)
  chunker.py        split text into overlapping chunks
  ollama_client.py  Ollama HTTP calls: embed(), chat_stream(), health()
  store.py          numpy vector store: cosine search, save/load, upsert
  pipeline.py       shared retrieval + prompt (used by ask.py and serve.py)
  indexer.py        ingest_paths(): load → chunk → embed → upsert → save
ingest.py           CLI: build/extend the index
ask.py              CLI: query the index (one-shot or REPL)
serve.py            web server (stdlib only): /api/status, /api/ingest, /api/ask
web/index.html      single-file UI (no build step, no CDN)
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

## Tuning

All knobs are env vars (see `rag/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_GEN_MODEL` | `llama3.2` | generation model (try `llama3.1:8b`, `qwen2.5`, `mistral`) |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `RAG_TEMPERATURE` | `0.2` | generation sampling temp — low = consistent, grounded answers (set `0` for fully deterministic) |
| `RAG_TOP_K` | `5` | chunks retrieved per question |
| `RAG_MIN_SCORE` | `0.6` | drop retrieved chunks below this cosine score (`0` = keep all); if nothing clears it, a corpus-overview fallback answers |
| `RAG_CHUNK_SIZE` | `900` | characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | overlap between chunks |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Changing the embedding model or chunk settings? Re-run `ingest.py` to rebuild the index.
