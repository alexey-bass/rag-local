"""Lexical retrieval — BM25 over chunk text — to complement dense embeddings.

Dense vectors capture *meaning* but blur exact tokens: a query for "Nokia", a
specific framework version, a job ID, or an uncommon acronym can rank a paraphrase
above the chunk that literally contains the term. BM25 is the opposite — it rewards
exact term overlap, weighted by how rare each term is and how long the chunk is.

We run both and fuse their rankings with Reciprocal Rank Fusion (RRF), which combines
ordered lists without needing their scores to be on the same scale. Pure stdlib + numpy,
so it adds no dependencies and stays Python-3.14-friendly.
"""
import math
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase word/number tokens. Deliberately simple — no stemming or stopwords."""
    return _TOKEN.findall((text or "").lower())


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents (record order preserved)."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        # Postings: term -> [(doc_index, term_frequency), ...]. Scoring then touches only
        # the documents that actually contain a query term, not the whole corpus.
        self.postings = {}
        for i, toks in enumerate(corpus_tokens):
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, f in counts.items():
                self.postings.setdefault(t, []).append((i, f))
        self.df = {t: len(plist) for t, plist in self.postings.items()}  # document frequency
        # IDF with the BM25 "+1" smoothing so it can't go negative for common terms.
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def discriminating(self, query_tokens, max_df_ratio=0.5):
        """Query terms worth a lexical vote: present in the corpus, but not in most of it.

        Drops terms absent from the index (no signal) and corpus-wide common terms
        (interrogatives, and "job"/"role"/"position" in a corpus of job posts), so BM25
        ranks on genuinely distinguishing tokens — names, IDs, versions — instead of noise.
        """
        cap = max_df_ratio * self.N
        return [t for t in query_tokens if 0 < self.df.get(t, 0) <= cap]

    def scores(self, query_tokens):
        """BM25 score of the query against every document (a length-N array, record order)."""
        out = np.zeros(self.N, dtype=np.float32)
        if not self.N or self.avgdl == 0:
            return out
        for t in set(query_tokens):
            plist = self.postings.get(t)
            idf = self.idf.get(t)
            if not plist or idf is None:
                continue
            for i, f in plist:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion of several ranked index lists → one fused index list.

    Each input is a list of document indices, best-first. An item's fused score is
    sum(1 / (k + rank)) across the lists it appears in, so agreement across retrievers
    wins and an item strong in just one list can still surface. `k` damps the influence
    of very top ranks; 60 is the value from the original RRF paper.
    """
    fused = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            idx = int(idx)
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda i: -fused[i])
