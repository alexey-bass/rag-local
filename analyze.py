#!/usr/bin/env python3
"""Analyze a document set and recommend a chunk size — before you spend time embedding.

Chunk size is a trade-off this script makes concrete for *your* docs:
  - too small  → one answer is scattered across chunks; with TOP_K retrieval you may
                 never see all the pieces, and the per-chunk context header is pure overhead.
  - too large  → a chunk's embedding averages several topics together, so it matches
                 everything weakly and ranks precisely for nothing.

It runs your files through the real loader + chunker (no embedding, no Ollama needed),
reports the length distribution, sweeps candidate sizes, and suggests one that keeps most
documents in just a few coherent chunks.

    .venv/bin/python analyze.py PATH...     # defaults to data/ if no path given
"""
import sys
from pathlib import Path

import numpy as np

from rag import config
from rag.chunker import _extract_context, chunk_text
from rag.loader import load_documents

SWEEP = [500, 700, 900, 1200, 1500, 1800, 2400, 3200]
OVERLAP_RATIO = 0.15  # overlap scales with size, so the sweep compares like with like


def _pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else 0.0


def _dist(label, arr):
    arr = np.array(arr, dtype=np.float64)
    print(f"  {label:18} min {int(arr.min()):>6}  median {int(np.median(arr)):>6}  "
          f"p90 {int(_pct(arr, 90)):>6}  max {int(arr.max()):>7}  mean {int(arr.mean()):>6}")


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("-")] or [str(config.DATA_DIR)]
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0

    docs = load_documents(paths)
    if not docs:
        print("No supported documents found.")
        return 1

    doc_lens, body_lens, header_lens = [], [], []
    for d in docs:
        ctx, body = _extract_context(d["text"], d["source"])
        doc_lens.append(len(d["text"]))
        body_lens.append(len(body))
        header_lens.append(len(ctx))

    print(f"\n{len(docs)} documents.  Lengths in characters:")
    _dist("whole document", doc_lens)
    _dist("body (chunked)", body_lens)
    _dist("context header", header_lens)
    print(f"\n  The context header is prepended to every chunk (median "
          f"{int(np.median(header_lens))} chars of overhead per chunk).")

    print(f"\nChunk-size sweep (overlap ≈ {int(OVERLAP_RATIO * 100)}% of size):\n")
    print(f"  {'size':>5} {'overlap':>7} {'chunks':>7} {'/doc med':>9} {'/doc max':>9} "
          f"{'1 chunk':>8} {'≤3 chunks':>10} {'avg fill':>9}")
    rows = []
    for size in SWEEP:
        overlap = round(size * OVERLAP_RATIO)
        per_doc, lengths = [], []
        for d in docs:
            cs = chunk_text(d["text"], size=size, overlap=overlap, source=d["source"])
            per_doc.append(len(cs))
            lengths.extend(len(c) for c in cs)
        per_doc = np.array(per_doc)
        total = int(per_doc.sum())
        one = (per_doc == 1).mean()
        le3 = (per_doc <= 3).mean()
        fill = np.mean(lengths) / size if lengths else 0
        rows.append((size, overlap, total, int(np.median(per_doc)), int(per_doc.max()), one, le3, fill))
        mark = "  <- current default" if size == config.CHUNK_SIZE else ""
        print(f"  {size:>5} {overlap:>7} {total:>7} {int(np.median(per_doc)):>9} "
              f"{int(per_doc.max()):>9} {one:>7.0%} {le3:>9.0%} {fill:>8.0%}{mark}")

    # Recommendation: smallest swept size that keeps the typical document to a few chunks
    # (median ≤ 2 and at least ~70% of docs in ≤3 chunks), so a post isn't shattered while
    # chunks stay topically tight. Fall back to the largest size if nothing qualifies.
    pick = next((r for r in rows if r[3] <= 2 and r[6] >= 0.70), rows[-1])
    size, overlap = pick[0], pick[1]
    print(f"\nRecommendation:  RAG_CHUNK_SIZE={size}  RAG_CHUNK_OVERLAP={overlap}")
    print(f"  → median {pick[3]} chunk(s)/doc, {pick[6]:.0%} of docs in ≤3 chunks, "
          f"avg chunk {pick[7]:.0%} full.")
    body_p90 = _pct(body_lens, 90)
    if size >= body_p90 * 1.2 and config.CHUNK_SIZE < size:
        print("  Most documents are short — a larger chunk keeps each post whole.")
    print("\nNext:  preview it    .venv/bin/python ingest.py --dry-run --chunk-size "
          f"{size} --chunk-overlap {overlap} " + " ".join(paths))
    print( "       then re-ingest  add --replace, and re-run eval.py to confirm the change helped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
