"""A minimal vector store: L2-normalized embeddings + cosine similarity search.

Backed by two files in index/:
  - embeddings.npy : the (N, dim) matrix
  - records.json   : the chunk text + metadata, parallel to the matrix rows
No database required — fine for tens of thousands of chunks on a laptop.
"""
import json
from pathlib import Path

import numpy as np

from . import config


def _normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorStore:
    def __init__(self, embeddings=None, records=None, embed_model=None):
        self.embeddings = np.zeros((0, 0), dtype=np.float32) if embeddings is None else embeddings
        self.records = records or []
        self.embed_model = embed_model

    def __len__(self):
        return len(self.records)

    def add(self, embeddings, records):
        embeddings = _normalize(embeddings)
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.records.extend(records)

    def remove_paths(self, paths):
        """Drop all chunks whose record['path'] is in `paths`. Used to upsert on re-ingest."""
        paths = set(paths)
        keep = [i for i, r in enumerate(self.records) if r.get("path") not in paths]
        removed = len(self.records) - len(keep)
        if removed:
            self.embeddings = self.embeddings[keep] if self.embeddings.size else self.embeddings
            self.records = [self.records[i] for i in keep]
        return removed

    def search(self, query_vec, k=None):
        """Return [(record, score), ...] for the top-k most similar chunks."""
        k = k or config.TOP_K
        if self.embeddings.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [(self.records[i], float(scores[i])) for i in top]

    def save(self, index_dir=None):
        d = Path(index_dir or config.INDEX_DIR)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        (d / "records.json").write_text(
            json.dumps({"embed_model": self.embed_model, "records": self.records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir=None):
        d = Path(index_dir or config.INDEX_DIR)
        emb, rec = d / "embeddings.npy", d / "records.json"
        if not emb.exists() or not rec.exists():
            raise FileNotFoundError(f"No index found in {d}. Build one with:  python ingest.py")
        meta = json.loads(rec.read_text(encoding="utf-8"))
        return cls(np.load(emb), meta["records"], meta.get("embed_model"))
