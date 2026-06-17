"""Shared RAG logic used by both the CLI (ask.py) and the web UI (serve.py).

Keeping the prompt + retrieval here means there's exactly one definition of how a
question becomes an answer, regardless of which front-end calls it.
"""
import numpy as np

from . import config
from .corpus import corpus_overview
from .lexical import BM25, rrf, tokenize
from .ollama_client import OllamaError, chat, embed

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


# ── conversational memory ────────────────────────────────────────────────────
# `history` is a list of {"question", "answer"} dicts, oldest first — the running
# conversation, carried by the front-end (browser thread / REPL loop), not the server.

def _recent(history):
    turns = [t for t in (history or []) if (t.get("question") or "").strip() and (t.get("answer") or "").strip()]
    return turns[-config.HISTORY_TURNS:]


def build_messages(question, hits, history=None):
    """Assemble the chat messages for a grounded answer: system + prior turns + this turn.

    Only the current turn carries the retrieved context block; prior turns are kept as plain
    Q&A so the model can resolve references ("it", "that role") without bloating the prompt.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in _recent(history):
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": build_user_message(question, hits)})
    return messages


def overview_messages(store, question):
    """Chat messages for the corpus-overview fallback (single turn — no history needed)."""
    return [
        {"role": "system", "content": OVERVIEW_PROMPT},
        {"role": "user", "content": overview_user_message(store, question)},
    ]


CONDENSE_PROMPT = (
    "You rewrite a follow-up question into a standalone one. Using the conversation so far, "
    "resolve any pronouns or ellipsis (it, that role, the company…) and output a single "
    "self-contained question that means the same thing on its own. Output ONLY the rewritten "
    "question — no preamble, no quotes. If it is already standalone, return it unchanged."
)


def condense_question(history, question):
    """Rewrite a follow-up into a standalone query using the conversation, for retrieval.

    No-ops (returns the question unchanged) when there's no usable history, when CONDENSE is
    off, or if the rewrite fails or looks unreasonable — retrieval then just uses the raw text.
    """
    turns = _recent(history)
    if not config.CONDENSE or not turns:
        return question
    convo = "\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in turns)
    user = f"Conversation so far:\n{convo}\n\nFollow-up question: {question}\n\nStandalone question:"
    try:
        out = chat([{"role": "system", "content": CONDENSE_PROMPT}, {"role": "user", "content": user}])
    except OllamaError:
        return question
    out = out.strip().strip('"').strip()
    return out if 0 < len(out) <= 400 else question
