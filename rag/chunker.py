"""Split a document into self-contained, retrieval-friendly chunks.

Two ideas make retrieval much better on structured docs (job posts, papers, etc.):

1. **Context anchoring** — every chunk is prefixed with a compact document header
   (title + leading metadata like Company/Location/Track). Without this, body chunks
   lose the document's identity and a query like "nokia devops" can't match the
   paragraph that describes the role but never names the company.

2. **Structure-/line-aware splitting** — we keep the front-matter intact, split the
   body on line boundaries (so bullet lists and sentences aren't cut mid-word), and
   overlap by whole lines. No more chunks that start with "...mation projects".
"""
import re

from . import config


def _clean(line):
    """Strip markdown bullet/heading markers and bold for the context header."""
    return line.lstrip("#-* ").replace("**", "").strip()


def _extract_context(text, source, cap=200):
    """Split a document into (context_header, body).

    context_header is a one-line summary built from the title + leading metadata,
    prepended to every chunk so each one carries the document's identity.
    """
    lines = text.splitlines()

    # Front-matter ends at the first "## " section header, if there is one.
    sec = next((i for i, l in enumerate(lines) if l.strip().startswith("## ")), None)
    if sec is not None:
        front, body_lines = lines[:sec], lines[sec:]
    else:
        # Otherwise take the leading H1 + contiguous bullet/metadata lines.
        j = 0
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("# ") or s.startswith(("- ", "* ")) or (s.startswith("**") and ":" in s) or not s:
                j += 1
            else:
                break
        front, body_lines = lines[:j], lines[j:]
        if not any(l.strip() for l in front):  # doc opens with prose -> no front-matter
            front, body_lines = [], lines

    parts = []
    for l in front:
        c = _clean(l)
        if not c or "http" in c.lower():  # skip blanks and URL-only metadata (noise)
            continue
        parts.append(c[:70].rsplit(" ", 1)[0] if len(c) > 70 else c)
    context = " · ".join(parts)
    if not context:  # fall back to a readable version of the file name
        stem = (source or "document").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        context = stem.replace("-", " ").replace("_", " ").strip()
    if len(context) > cap:
        context = context[:cap].rsplit(" ", 1)[0] + "…"

    return context, "\n".join(body_lines).strip()


def _space_split(text, size, overlap):
    """Last-resort word-safe split for a single oversized line."""
    out, words, cur = [], text.split(" "), ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = (cur[-overlap:].split(" ", 1)[-1] + " " + w).strip()
        else:
            cur = f"{cur} {w}".strip() if cur else w
    if cur:
        out.append(cur)
    return out


def _split_body(body, budget, overlap):
    """Pack body lines into chunks <= budget, overlapping by whole trailing lines."""
    lines, chunks, cur, cur_len = body.split("\n"), [], [], 0
    for ln in lines:
        if cur and cur_len + len(ln) + 1 > budget:
            chunks.append("\n".join(cur).strip())
            keep, acc = [], 0  # carry trailing lines as overlap
            for prev in reversed(cur):
                if keep and acc + len(prev) + 1 > overlap:
                    break
                keep.insert(0, prev)
                acc += len(prev) + 1
            cur, cur_len = keep[:], sum(len(x) + 1 for x in keep)
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur).strip())

    out = []
    for c in chunks:  # hard-split any monster single line
        out.extend([c] if len(c) <= budget * 1.5 else _space_split(c, budget, overlap))
    return [c for c in out if c.strip()]


def chunk_text(text, size=None, overlap=None, source=None):
    """Chunk a document. Every chunk is prefixed with the document's context header."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = (text or "").strip()
    if not text:
        return []

    context, body = _extract_context(text, source)
    budget = max(350, size - len(context) - 2)  # leave room for the prefix
    body_chunks = _split_body(body, budget, overlap) if body else []
    if not body_chunks:
        return [context] if context else []
    return [f"{context}\n\n{bc}" for bc in body_chunks]
