#!/usr/bin/env python3
"""Analyze a document set and recommend a chunk size — before you spend time embedding.

Runs your files through the real loader + chunker (no embedding, no Ollama), reports the
length distribution, sweeps candidate sizes, and suggests one that keeps most documents in
just a few coherent chunks. The same analysis backs the web UI's "Analyze" button.

    .venv/bin/python analyze.py PATH...     # defaults to data/ if no path given
"""
import sys

from rag import config
from rag.analysis import OVERLAP_RATIO, analyze_paths


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("-")] or [str(config.DATA_DIR)]
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0

    r = analyze_paths(paths)
    if not r["documents"]:
        print("No supported documents found.")
        return 1

    L = r["lengths"]
    print(f"\n{r['documents']} documents.  Lengths in characters:")
    for label, key in (("whole document", "document"), ("body (chunked)", "body")):
        s = L[key]
        print(f"  {label:18} min {s['min']:>6}  median {s['median']:>6}  "
              f"p90 {s['p90']:>6}  max {s['max']:>7}  mean {s['mean']:>6}")
    print(f"\n  The context header is prepended to every chunk "
          f"(median {r['header_median']} chars of overhead per chunk).")

    print(f"\nChunk-size sweep (overlap ≈ {int(OVERLAP_RATIO * 100)}% of size):\n")
    print(f"  {'size':>5} {'overlap':>7} {'chunks':>7} {'/doc med':>9} {'/doc max':>9} "
          f"{'1 chunk':>8} {'≤3 chunks':>10} {'avg fill':>9}")
    for row in r["sweep"]:
        mark = "  <- current default" if row["size"] == r["current_default"] else ""
        print(f"  {row['size']:>5} {row['overlap']:>7} {row['chunks']:>7} {row['per_doc_median']:>9} "
              f"{row['per_doc_max']:>9} {row['pct_one']:>7.0%} {row['pct_le3']:>9.0%} {row['fill']:>8.0%}{mark}")

    rec = r["recommended"]
    print(f"\nRecommendation:  RAG_CHUNK_SIZE={rec['chunk_size']}  RAG_CHUNK_OVERLAP={rec['chunk_overlap']}")
    print(f"  → median {rec['per_doc_median']} chunk(s)/doc, {rec['pct_le3']:.0%} of docs in ≤3 chunks.")
    print("\nNext:  preview it    .venv/bin/python ingest.py --dry-run --chunk-size "
          f"{rec['chunk_size']} --chunk-overlap {rec['chunk_overlap']} " + " ".join(paths))
    print( "       then re-ingest  add --replace, and re-run eval.py to confirm the change helped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
