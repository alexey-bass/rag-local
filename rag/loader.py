"""Load raw text from files. Accepts any mix of file and directory paths;
directories are searched recursively. Supports .txt, .md/.markdown, and .pdf.

Each returned doc carries:
  - path   : absolute path on disk (used to upsert/dedupe on re-ingest)
  - source : a short display label (relative to the pasted folder, or the filename)
  - text   : the extracted text
"""
from pathlib import Path

from . import config

TEXT_EXTS = {".txt", ".md", ".markdown", ".text"}
PDF_EXTS = {".pdf"}
SUPPORTED = TEXT_EXTS | PDF_EXTS


def _read_pdf(path):
    from pypdf import PdfReader  # lazy import so non-PDF runs don't load it

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _extract(path):
    if path.suffix.lower() in PDF_EXTS:
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_paths(paths):
    """Accept None, a single str/Path, or a list. Strip stray quotes/whitespace."""
    if paths is None:
        return [config.DATA_DIR]
    if isinstance(paths, (str, Path)):
        paths = [paths]
    cleaned = []
    for p in paths:
        s = str(p).strip().strip('"').strip("'").strip()
        if s:
            cleaned.append(s)
    return cleaned


def _files_under(root):
    """Yield (file_path, display_label) for one pasted path (file or dir)."""
    root = Path(root).expanduser()
    if root.is_file():
        yield root, root.name
    elif root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                yield p, str(p.relative_to(root))
    else:
        print(f"  ! not found: {root}")


def load_documents(paths=None):
    """Return a list of {path, source, text} for all supported files under `paths`.

    `paths` may be None (defaults to data/), a single path, or a list of paths.
    """
    docs, seen = [], set()
    for root in _normalize_paths(paths):
        for fpath, display in _files_under(root):
            if fpath.suffix.lower() not in SUPPORTED:
                continue
            abspath = str(fpath.resolve())
            if abspath in seen:  # same file reached via two overlapping roots
                continue
            seen.add(abspath)
            try:
                text = _extract(fpath)
            except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
                print(f"  ! skipping {fpath.name}: {e}")
                continue
            text = text.strip()
            if text:
                docs.append({"path": abspath, "source": display, "text": text})
    return docs
