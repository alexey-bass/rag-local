"""Profile a document set and recommend a chunk size — without embedding anything.

Chunk size is a trade-off: too small scatters one answer across chunks (and the per-chunk
context header becomes pure overhead); too large averages several topics into one vector
that matches everything weakly. This module runs files through the real loader + chunker,
reports the length distribution, and sweeps candidate sizes so the choice is data-driven.

**Shared** by `analyze.py` (CLI) and `serve.py` (the UI's /api/analyze). Returns plain dicts
so it serializes straight to JSON. No Ollama, no embedding.
"""
import numpy as np

from . import config
from .chunker import _extract_context, chunk_text
from .loader import load_documents

SWEEP = [500, 700, 900, 1200, 1500, 1800, 2400, 3200]
OVERLAP_RATIO = 0.15  # overlap scales with size, so the sweep compares like with like


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": int(a.min()), "median": int(np.median(a)),
        "p90": int(np.percentile(a, 90)), "max": int(a.max()), "mean": int(a.mean()),
    }


def analyze_paths(paths, sweep=SWEEP, overlap_ratio=OVERLAP_RATIO):
    """Profile the docs under `paths` and recommend a chunk size.

    Returns {documents, lengths{document,body}, header_median, sweep[...], recommended, current_default}.
    `documents` is 0 (and sweep empty) when nothing supported is found.
    """
    docs = load_documents(paths)
    if not docs:
        return {"documents": 0, "lengths": None, "header_median": 0,
                "sweep": [], "recommended": None, "current_default": config.CHUNK_SIZE}

    doc_lens, body_lens, header_lens = [], [], []
    for d in docs:
        ctx, body = _extract_context(d["text"], d["source"])
        doc_lens.append(len(d["text"]))
        body_lens.append(len(body))
        header_lens.append(len(ctx))

    rows = []
    for size in sweep:
        overlap = round(size * overlap_ratio)
        per_doc, lengths = [], []
        for d in docs:
            cs = chunk_text(d["text"], size=size, overlap=overlap, source=d["source"])
            per_doc.append(len(cs))
            lengths.extend(len(c) for c in cs)
        pd = np.asarray(per_doc)
        rows.append({
            "size": size, "overlap": overlap, "chunks": int(pd.sum()),
            "per_doc_median": int(np.median(pd)), "per_doc_max": int(pd.max()),
            "pct_one": float((pd == 1).mean()), "pct_le3": float((pd <= 3).mean()),
            "fill": float(np.mean(lengths) / size) if lengths else 0.0,
        })

    # Recommend the smallest swept size that keeps the typical document to a few chunks
    # (median ≤ 2 and ≥70% of docs in ≤3 chunks), so a doc isn't shattered while chunks
    # stay topically tight. Fall back to the largest size if nothing qualifies.
    pick = next((r for r in rows if r["per_doc_median"] <= 2 and r["pct_le3"] >= 0.70), rows[-1])
    return {
        "documents": len(docs),
        "lengths": {"document": _stats(doc_lens), "body": _stats(body_lens)},
        "header_median": int(np.median(header_lens)),
        "sweep": rows,
        "recommended": {
            "chunk_size": pick["size"], "chunk_overlap": pick["overlap"],
            "per_doc_median": pick["per_doc_median"], "pct_le3": pick["pct_le3"],
        },
        "current_default": config.CHUNK_SIZE,
    }
