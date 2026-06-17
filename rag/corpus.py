"""Corpus-level facts for aggregate / meta questions.

Semantic retrieval finds *similar passages* — it can't count documents or list
companies, because no single passage holds that answer and the model only ever
sees the top-k matches. So when a question matches no passage, we fall back to
this compact, computed overview, letting questions like "how many docs do we
have?" or "what companies are indexed?" still be answered factually.
"""
import re
from collections import Counter

_COMPANY = re.compile(r"Company:\s*([^·\n]+)")


def corpus_overview(store, top=12):
    """Return a short, factual summary of the indexed collection."""
    # The lowest-numbered chunk of each source carries the metadata prefix.
    first = {}
    for r in store.records:
        s = r["source"]
        if s not in first or r["chunk"] < first[s]["chunk"]:
            first[s] = r

    companies = Counter()
    for r in first.values():
        m = _COMPANY.search(r["text"])
        if m:
            companies[m.group(1).strip()] += 1

    lines = [f"The indexed collection has {len(first)} documents ({len(store.records)} chunks)."]
    if companies:
        lines.append(f"It spans {len(companies)} distinct companies.")
        top_list = "; ".join(f"{c} ({n})" for c, n in companies.most_common(top))
        lines.append(f"Companies with the most documents: {top_list}.")
    return "\n".join(lines)
