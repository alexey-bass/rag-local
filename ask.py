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
from rag.pipeline import build_messages, condense_question, overview_messages, retrieve
from rag.store import VectorStore


def answer(store, question, history=None):
    """Answer a question (streamed to stdout) and return the full answer text.

    `history` is the running list of {question, answer} turns (REPL only); it lets a
    follow-up resolve against earlier ones. Returns the answer so the caller can append it.
    """
    if len(store) == 0:
        print("The index is empty. Run `python ingest.py` first.")
        return ""

    standalone = condense_question(history, question)
    hits = retrieve(store, standalone)
    text = ""
    print()
    if not hits:
        # No passage matched — answer corpus-level questions from a computed overview.
        for piece in chat_stream(overview_messages(store, standalone)):
            print(piece, end="", flush=True)
            text += piece
        print("\n")
        return text

    for piece in chat_stream(build_messages(question, hits, history)):
        print(piece, end="", flush=True)
        text += piece
    print("\n")

    sources = []
    for rec, _ in hits:
        tag = f"{rec['source']}#{rec['chunk']}"
        if tag not in sources:
            sources.append(tag)
    print("Sources: " + ", ".join(sources))
    return text


def main():
    try:
        store = VectorStore.load()
    except FileNotFoundError as e:
        print(e)
        return 1

    print(f"Loaded index: {len(store)} chunks  |  gen model: {config.GEN_MODEL}")

    if len(sys.argv) > 1:
        answer(store, " ".join(sys.argv[1:]))  # one-shot: no conversation
        return 0

    print('Ask a question (follow-ups welcome · "new" clears the thread · "exit" to quit).')
    history = []  # running conversation, so follow-ups resolve against earlier turns
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
        if q.lower() in {"new", "reset", "clear"}:
            history = []
            print("(new conversation)")
            continue
        text = answer(store, q, history)
        history.append({"question": q, "answer": text})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OllamaError as e:
        print(f"\n[Ollama] {e}", file=sys.stderr)
        sys.exit(2)
