"""Shared RAG logic used by both the CLI (ask.py) and the web UI (serve.py).

Keeping the prompt + retrieval here means there's exactly one definition of how a
question becomes an answer, regardless of which front-end calls it.
"""
import numpy as np

from . import config
from .corpus import corpus_overview
from .lexical import BM25, rrf, tokenize
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


def _bm25_for(store):
    """Lazily build (and cache on the store) a BM25 index over the chunk texts.

    Rebuilt whenever the store's chunk count changes, so re-ingesting refreshes it.
    The cache lives on the store instance, which both front-ends keep alive across
    questions (serve.py caches it; ask.py loads it once per process).
    """
    n = len(store)
    if getattr(store, "_bm25", None) is None or getattr(store, "_bm25_n", None) != n:
        store._bm25 = BM25([tokenize(r["text"]) for r in store.records])
        store._bm25_n = n
    return store._bm25


def retrieve(store, question, k=None, min_score=None):
    """Retrieve the most relevant [(record, cosine_score), ...] passages for a question.

    Pipeline:
      1. Embed the question (with the query-side task prefix) and score it against
         every chunk by cosine similarity.
      2. Off-topic gate: if even the best chunk is below `min_score`, return [] so the
         front-ends fall back to the computed corpus overview.
      3. If hybrid retrieval is on, fuse the dense ranking with a BM25 lexical ranking
         via RRF — this pulls exact-token matches (company names, IDs, versions) up.
      4. Trim the tail relative to the top hit (drop chunks below
         config.MIN_SCORE_RATIO * top_score) and return up to k passages.

    Scores returned are always the cosine similarity, so the UI's "similarity" stays meaningful.
    """
    k = k or config.TOP_K
    if min_score is None:
        min_score = config.MIN_SCORE

    query_vec = embed(question, prefix=config.EMBED_QUERY_PREFIX)[0]
    dense = store.scores(query_vec)
    if dense.size == 0:
        return []

    order = np.argsort(-dense)
    top_score = float(dense[order[0]])
    if top_score < min_score:  # nothing is even plausibly on-topic -> overview fallback
        return []

    pool = max(config.RETRIEVE_POOL, k)
    dense_pool = [int(i) for i in order[:pool]]

    if config.HYBRID:
        bm = _bm25_for(store)
        terms = bm.discriminating(tokenize(question), config.LEXICAL_MAX_DF_RATIO)
        lex_pool = []
        if terms:  # only fuse when the query has a distinguishing token to match on
            lex = bm.scores(terms)
            lex_pool = [int(i) for i in np.argsort(-lex)[:pool] if lex[i] > 0]
        ranked = rrf([dense_pool, lex_pool]) if lex_pool else dense_pool
    else:
        ranked = dense_pool

    floor = config.MIN_SCORE_RATIO * top_score
    hits = []
    for i in ranked:
        score = float(dense[i])
        if score >= floor:  # keep the cosine floor so BM25 can't smuggle in off-topic chunks
            hits.append((store.records[i], score))
        if len(hits) >= k:
            break
    return hits


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
