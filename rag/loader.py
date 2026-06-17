"""Load raw text from files. Accepts any mix of file and directory paths;
directories are searched recursively. Supports .txt, .md/.markdown, and .pdf.

Each returned doc carries:
  - path   : absolute path on disk (used to upsert/dedupe on re-ingest)
  - source : a short display label (relative to the pasted folder, or the filename)
  - text   : the extracted text
"""
import glob as globlib
from pathlib import Path

from . import config

_GLOB_CHARS = "*?["

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


def _has_glob(s):
    return any(c in s for c in _GLOB_CHARS)


def _glob_base(pattern):
    """The leading path components of `pattern` before the first wildcard — the root that
    matched files' display labels are made relative to (so labels stay short and meaningful)."""
    parts = []
    for part in Path(pattern).parts:
        if _has_glob(part):
            break
        parts.append(part)
    return Path(*parts) if parts else Path(".")


def _label(path, base):
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def _walk_dir(directory, base):
    """Yield (file, label) for every file under `directory` (recursive). Read-only; unreadable
    directories/entries are skipped rather than raising, so one bad folder can't abort the scan."""
    try:
        entries = sorted(directory.rglob("*"))
    except OSError as e:
        print(f"  ! skipping {directory}: {e}")
        return
    for p in entries:
        try:
            if p.is_file():
                yield p, _label(p, base)
        except OSError:
            continue


def _files_under(root):
    """Yield (file_path, display_label) for one pasted path: a file, a directory (recursive),
    or a glob pattern. A glob may match files and/or directories — matched directories are
    walked recursively:
        ~/papers/*.pdf                      → matching files
        .../offers/2606*                    → every file inside each matching folder
        ~/docs/**/*.md                      → recursive match
    Expansion is local and read-only; results are still filtered to supported types upstream."""
    expanded = Path(root).expanduser()
    if expanded.is_file():
        yield expanded, expanded.name
    elif expanded.is_dir():
        yield from _walk_dir(expanded, expanded)
    elif _has_glob(str(expanded)):
        pattern = str(expanded)
        base = _glob_base(pattern)
        matched = False
        for m in sorted(globlib.glob(pattern, recursive=True)):
            mp = Path(m)
            if mp.is_dir():
                matched = True
                yield from _walk_dir(mp, base)
            elif mp.is_file():
                matched = True
                yield mp, _label(mp, base)
        if not matched:
            print(f"  ! no files match: {root}")
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
