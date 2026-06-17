"""Central configuration. Everything here can be overridden via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"      # you drop your .txt/.md/.pdf files here
INDEX_DIR = ROOT / "index"    # the built vector index lives here

# Ollama server + models. Override with RAG_GEN_MODEL / RAG_EMBED_MODEL / OLLAMA_HOST.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "llama3.2")

# Embedding task prefixes. nomic-embed-text is an ASYMMETRIC model: it was trained with
# a task instruction on every input, and the document side and query side use *different*
# prefixes so they land in aligned regions of the space. Omitting them runs the model
# out-of-distribution and measurably hurts retrieval. Stored documents get DOC_PREFIX at
# index time; questions get QUERY_PREFIX at ask time. Set both to "" for a model that
# doesn't want prefixes (e.g. bge-m3). Changing either invalidates the index — re-ingest.
EMBED_DOC_PREFIX = os.environ.get("RAG_EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.environ.get("RAG_EMBED_QUERY_PREFIX", "search_query: ")

# Chunking + retrieval knobs.
# Defaults are tuned to the bundled corpus (short, self-contained docs like job posts):
# a measured sweep with eval.py found 3200 best (whole doc per chunk → highest recall@5 +
# rank quality + top-k document diversity). For long-form docs, re-tune with analyze.py.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "3200"))     # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "480"))  # overlap between chunks
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))                  # chunks returned per question
GEN_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.2"))  # low = consistent, grounded answers

# Conversational memory. The front-ends carry the running conversation (the browser thread,
# the REPL loop) and replay the last HISTORY_TURNS exchanges so follow-ups stay in context.
# CONDENSE rewrites a follow-up into a standalone query *before* retrieval (one extra LLM call)
# so "what about its salary?" still retrieves the right chunks; set RAG_CONDENSE=0 to skip it.
HISTORY_TURNS = int(os.environ.get("RAG_HISTORY_TURNS", "6"))       # prior Q&A exchanges kept as context
CONDENSE = os.environ.get("RAG_CONDENSE", "1").lower() not in ("0", "false", "no", "")

# Hybrid retrieval: blend dense (embedding) ranking with lexical BM25 ranking via
# Reciprocal Rank Fusion. Dense captures meaning; BM25 nails exact tokens — company
# names, locations, tech/version strings — where embeddings are weakest. RAG_HYBRID=0
# disables it (pure dense). RETRIEVE_POOL is how many candidates each retriever
# contributes before fusion (must be >= TOP_K).
HYBRID = os.environ.get("RAG_HYBRID", "1").lower() not in ("0", "false", "no", "")
RETRIEVE_POOL = int(os.environ.get("RAG_RETRIEVE_POOL", "20"))
# A query term only earns a BM25 vote if it appears in at most this fraction of all
# chunks. Filters corpus-wide noise ("the", "job", "position") so lexical matching
# fires on distinguishing tokens — names, IDs, versions.
LEXICAL_MAX_DF_RATIO = float(os.environ.get("RAG_LEXICAL_MAX_DF_RATIO", "0.5"))

# Score gating. MIN_SCORE is an absolute cosine floor on the *best* hit: if even the top
# match is below it, the question is treated as off-topic and the front-ends fall back to
# the computed corpus overview (so "how many docs?" still works). MIN_SCORE_RATIO then
# trims the tail *relative* to the top hit (keep hits >= ratio * top_score), which is far
# more robust across corpora than a second fixed constant.
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.6"))           # off-topic gate (0 = never fall back)
MIN_SCORE_RATIO = float(os.environ.get("RAG_MIN_SCORE_RATIO", "0.5"))  # tail floor as a fraction of the top hit
