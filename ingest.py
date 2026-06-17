#!/usr/bin/env python3
"""Build (or extend) the vector index.

Ingest the default data/ folder:
    python ingest.py

Ingest any file or folder you point at (folders are searched recursively):
    python ingest.py /Users/alexey/Documents/notes
    python ingest.py ~/papers/paper.pdf ~/wiki

Re-ingesting a path updates its chunks (no duplicates). Use --replace to wipe
the whole index and start fresh:
    python ingest.py --replace ~/papers

Options:
    --replace              wipe the index and rebuild from the given paths
    --dry-run              scan + chunk and report counts, without embedding or saving
    --chunk-size N         override chunk size (chars) for this run
    --chunk-overlap N      override chunk overlap (chars) for this run
"""
import sys

from rag import config
from rag.indexer import ingest_paths
from rag.ollama_client import OllamaError


def _progress(event):
    t = event["type"]
    if t == "status":
        print(event["message"])
    elif t == "loaded":
        print(f"  {event['documents']} document(s) found.")
    elif t == "chunked":
        print(f"  {event['chunks']} chunk(s) to embed (model: {config.EMBED_MODEL}).")
    elif t == "embed":
        print(f"  embedded {event['current']}/{event['total']}", end="\r")


def main():
    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0

    replace = dry_run = False
    chunk_size = chunk_overlap = None
    paths = []
    it = iter(argv)
    for a in it:
        if a in ("--replace", "-r"):
            replace = True
        elif a == "--dry-run":
            dry_run = True
        elif a in ("--chunk-size", "--chunk-overlap"):
            try:
                value = int(next(it))
            except (StopIteration, ValueError):
                print(f"[error] {a} needs an integer", file=sys.stderr)
                return 2
            chunk_size, chunk_overlap = (value, chunk_overlap) if a == "--chunk-size" else (chunk_size, value)
        elif a.startswith("-"):
            print(f"[error] unknown option {a}", file=sys.stderr)
            return 2
        else:
            paths.append(a)

    paths = paths or [str(config.DATA_DIR)]
    stats = ingest_paths(paths, replace=replace, emit=_progress,
                         chunk_size=chunk_size, chunk_overlap=chunk_overlap, dry_run=dry_run)

    if not stats["documents"]:
        print(
            f"\nNo supported files (.txt/.md/.pdf) found in: {', '.join(paths)}\n"
            "Check the path, or drop files into data/ and run `python ingest.py`."
        )
        return 1

    if stats.get("dry_run"):
        print(f"\n[dry run] {stats['documents']} file(s) → ~{stats['chunks']} chunks. "
              "Nothing was embedded or saved.")
        return 0

    print(f"\nDone. Index now holds {stats['total_chunks']} chunks (from {stats['documents']} file(s)).")
    print('Ask it something:  python ask.py "your question here"')
    print("Or open the web UI:  python serve.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OllamaError, RuntimeError) as e:
        print(f"\n[error] {e}", file=sys.stderr)
        sys.exit(2)
