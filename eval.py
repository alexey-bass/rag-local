#!/usr/bin/env python3
"""Tiny retrieval-quality harness — measure, don't guess.

Every chunking/embedding knob (chunk size, task prefixes, hybrid on/off, the score
floor, the embed model) is a trade-off you can only judge against a fixed set of
questions with known-good sources. This script runs your questions through the real
`retrieve()` pipeline and reports how often the expected source comes back, so you can
A/B a change instead of eyeballing one answer.

    .venv/bin/python eval.py                       # uses eval/questions.json
    .venv/bin/python eval.py path/to/questions.json
    .venv/bin/python eval.py --k 8                 # retrieve 8 instead of TOP_K

Compare configurations by toggling env vars in front of it (each run re-embeds the
queries; the index itself only changes when you re-ingest):

    RAG_HYBRID=0 .venv/bin/python eval.py          # dense only
    RAG_HYBRID=1 .venv/bin/python eval.py          # dense + BM25 (default)

questions.json is a list of {"q": <question>, "expect": <substring of the source label>}.
A question "hits" when any retrieved passage's source contains `expect` (case-insensitive).
Retrieval only — no generation — so it's fast and needs only the embedding model.
"""
import json
import sys

from rag import config
from rag.ollama_client import OllamaError
from rag.pipeline import retrieve
from rag.store import VectorStore

DEFAULT_QUESTIONS = config.ROOT / "eval" / "questions.json"


def _load_questions(path):
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must be a non-empty JSON list of {{'q','expect'}} objects.")
    return items


def main():
    args = sys.argv[1:]
    k = config.TOP_K
    path = DEFAULT_QUESTIONS
    it = iter(args)
    for a in it:
        if a == "--k":
            k = int(next(it))
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        elif not a.startswith("-"):
            from pathlib import Path
            path = Path(a)

    if not path.exists():
        print(f"No questions file at {path}.\n"
              f"Create a JSON list like: [{{\"q\": \"What is the Capco role?\", \"expect\": \"capco\"}}]")
        return 1

    try:
        store = VectorStore.load()
    except FileNotFoundError as e:
        print(e)
        return 1
    questions = _load_questions(path)

    mode = "dense+BM25 (hybrid)" if config.HYBRID else "dense only"
    print(f"Index: {len(store)} chunks | retrieve k={k} | {mode} | "
          f"min_score={config.MIN_SCORE} ratio={config.MIN_SCORE_RATIO}\n")

    hits = 0
    rr_sum = 0.0          # for mean reciprocal rank
    top_scores = []
    for item in questions:
        q, expect = item["q"], item["expect"].lower()
        results = retrieve(store, q, k=k)
        rank = next((i for i, (rec, _) in enumerate(results, 1)
                     if expect in rec["source"].lower()), None)
        if results:
            top_scores.append(results[0][1])
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        if not results:
            mark, tail = "— fell back", "(no passage cleared the gate)"
        else:
            mark = f"hit @{rank}" if rank else "miss"
            tail = f"top={results[0][0]['source']} ({results[0][1]:.3f})"
        print(f"  {mark:<12} expect~{item['expect']:<14} {tail}")

    n = len(questions)
    mean_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    print(f"\nrecall@{k}: {hits}/{n} = {hits / n:.0%}   "
          f"MRR: {rr_sum / n:.3f}   mean top-score: {mean_top:.3f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OllamaError as e:
        print(f"\n[Ollama] {e}", file=sys.stderr)
        sys.exit(2)
