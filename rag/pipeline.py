"""Shared RAG logic used by both the CLI (ask.py) and the web UI (serve.py).

Keeping the prompt + retrieval here means there's exactly one definition of how a
question becomes an answer, regardless of which front-end calls it.
"""
from . import config
from .corpus import corpus_overview
from .ollama_client import embed

SYSTEM_PROMPT = (
    "You answer questions about the user's own documents, using the numbered context "
    "passages below. The passages were retrieved as the most relevant matches for the "
    "question, and their headers and metadata (file name, company, title, location, etc.) "
    "are meaningful content you can rely on.\n"
    "- Answer directly and concisely, and cite the passages you use like [1], [2].\n"
    "- Ground every claim in the passages; don't invent details they don't support.\n"
    "- Only say you don't know if the passages genuinely contain nothing relevant."
)


def retrieve(store, question, k=None, min_score=None):
    """Embed the question and return the top-k [(record, score), ...] hits.

    Hits scoring below `min_score` (cosine similarity, defaults to
    config.MIN_SCORE) are dropped, so an off-topic question yields few or zero
    passages instead of filler near-misses. Pass min_score=0 to keep all top-k.
    """
    if min_score is None:
        min_score = config.MIN_SCORE
    query_vec = embed(question)[0]
    hits = store.search(query_vec, k=k)
    return [(rec, score) for rec, score in hits if score >= min_score]


def build_user_message(question, hits):
    """Format retrieved chunks into a numbered context block for the model."""
    blocks = []
    for n, (rec, score) in enumerate(hits, start=1):
        blocks.append(f"[{n}] (from {rec['source']}, similarity {score:.2f})\n{rec['text']}")
    context = "\n\n".join(blocks)
    return f"Context passages:\n\n{context}\n\n---\nQuestion: {question}"


OVERVIEW_PROMPT = (
    "The user is asking about their indexed document collection, but no individual passage "
    "matched the question. Use the collection overview below to answer questions about the "
    "collection's size or composition (document counts, companies, etc.), quoting its numbers. "
    "If the question needs specifics not present in the overview, say no matching document was found."
)


def overview_user_message(store, question):
    """Fallback context for corpus-level questions when no passage clears the score floor."""
    return f"Collection overview:\n{corpus_overview(store)}\n\n---\nQuestion: {question}"
