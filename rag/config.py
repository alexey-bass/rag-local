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

# Chunking + retrieval knobs.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "900"))      # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "150"))  # overlap between chunks
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))                  # chunks retrieved per question
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.6"))      # drop hits below this cosine score (0 = keep all)
GEN_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.2"))  # low = consistent, grounded answers
