#!/usr/bin/env python3
"""Ask questions against the index built by ingest.py.

One-shot:
    python ask.py "what does the document say about X?"

Interactive (REPL):
    python ask.py
"""
import sys

from rag import config
from rag.ollama_client import OllamaError, chat_stream
from rag.pipeline import SYSTEM_PROMPT, build_user_message, retrieve
from rag.store import VectorStore


def answer(store, question):
    if len(store) == 0:
        print("The index is empty. Run `python ingest.py` first.")
        return
    hits = retrieve(store, question)
    if not hits:
        print("\nI couldn't find anything relevant to that in your indexed documents.\n")
        return

    user_msg = build_user_message(question, hits)
    print()
    for piece in chat_stream(SYSTEM_PROMPT, user_msg):
        print(piece, end="", flush=True)
    print("\n")

    sources = []
    for rec, _ in hits:
        tag = f"{rec['source']}#{rec['chunk']}"
        if tag not in sources:
            sources.append(tag)
    print("Sources: " + ", ".join(sources))


def main():
    try:
        store = VectorStore.load()
    except FileNotFoundError as e:
        print(e)
        return 1

    print(f"Loaded index: {len(store)} chunks  |  gen model: {config.GEN_MODEL}")

    if len(sys.argv) > 1:
        answer(store, " ".join(sys.argv[1:]))
        return 0

    print('Ask a question (or "exit" to quit).')
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        answer(store, q)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OllamaError as e:
        print(f"\n[Ollama] {e}", file=sys.stderr)
        sys.exit(2)
