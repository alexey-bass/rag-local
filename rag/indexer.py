"""Reusable ingestion: turn files/folders into index chunks.

Both ingest.py (CLI) and serve.py (web) call ingest_paths(); each just formats
the emitted events differently (terminal prints vs NDJSON to the browser).

Upsert semantics: re-ingesting a file replaces its old chunks instead of
duplicating them, so you can paste the same folder again after editing it.
"""
from . import config
from .chunker import chunk_text
from .loader import load_documents
from .ollama_client import embed
from .store import VectorStore

BATCH = 32  # chunks per Ollama embed request


def _noop(_event):
    pass


def _existing_size():
    try:
        return len(VectorStore.load())
    except FileNotFoundError:
        return 0


def ingest_paths(paths, replace=False, emit=_noop, chunk_size=None, chunk_overlap=None, dry_run=False):
    """Ingest file/dir paths (recursive) into the index.

    chunk_size / chunk_overlap override the defaults for this run (None = config default).
    dry_run scans + chunks and reports what *would* be indexed, without embedding or saving.
    emit(event) is called with dicts like {"type": "embed", "current": 64, "total": 200}.
    Returns a stats dict.
    """
    emit({"type": "status", "message": f"Scanning {paths} ..."})
    docs = load_documents(paths)
    emit({"type": "loaded", "documents": len(docs), "files": [d["source"] for d in docs]})
    if not docs:
        return {"documents": 0, "chunks": 0, "total_chunks": _existing_size(), "files": [], "dry_run": dry_run}

    # Chunk every document — needed both for a dry-run estimate and a real ingest.
    records, texts = [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], size=chunk_size, overlap=chunk_overlap, source=doc["source"])):
            records.append({"source": doc["source"], "path": doc["path"], "chunk": i, "text": chunk})
            texts.append(chunk)
    emit({"type": "chunked", "chunks": len(texts)})

    if dry_run:  # preview only — don't embed or touch the index
        stats = {"documents": len(docs), "chunks": len(texts), "total_chunks": _existing_size(),
                 "files": [d["source"] for d in docs], "dry_run": True}
        emit({"type": "done", **stats})
        return stats

    # Start fresh, or load the existing index to add to.
    if replace:
        store = VectorStore(embed_model=config.EMBED_MODEL)
    else:
        try:
            store = VectorStore.load()
        except FileNotFoundError:
            store = VectorStore(embed_model=config.EMBED_MODEL)
        if len(store) and store.embed_model and store.embed_model != config.EMBED_MODEL:
            raise RuntimeError(
                f"Existing index was built with embedding model '{store.embed_model}', but "
                f"current model is '{config.EMBED_MODEL}'. Re-ingest with replace to rebuild."
            )

    # Upsert: clear any prior chunks for the files we're about to (re)ingest.
    removed = store.remove_paths(d["path"] for d in docs)
    if removed:
        emit({"type": "status", "message": f"Replacing {removed} existing chunk(s) for these files"})

    for start in range(0, len(texts), BATCH):
        vectors = embed(texts[start : start + BATCH])
        store.add(vectors, records[start : start + BATCH])
        emit({"type": "embed", "current": min(start + BATCH, len(texts)), "total": len(texts)})

    store.embed_model = config.EMBED_MODEL
    store.save()
    stats = {
        "documents": len(docs),
        "chunks": len(texts),
        "total_chunks": len(store),
        "files": [d["source"] for d in docs],
        "dry_run": False,
    }
    emit({"type": "done", **stats})
    return stats
