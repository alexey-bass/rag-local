#!/usr/bin/env python3
"""Local web UI for the RAG system. No extra dependencies — Python stdlib only.

    python serve.py            # http://127.0.0.1:8000
    python serve.py 8080       # custom port

Open the URL in a browser: paste a path to ingest (recursive), then ask questions
and watch answers stream in with cited, expandable source snippets. Everything
stays on your machine — it binds to 127.0.0.1 only.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag import config
from rag.analysis import analyze_paths
from rag.indexer import ingest_paths
from rag.ollama_client import OllamaError, chat_stream, health
from rag.pipeline import build_messages, condense_question, overview_messages, retrieve
from rag.store import VectorStore

WEB_DIR = config.ROOT / "web"


def log(msg):
    """Print an activity line to the server's stdout (visible in the terminal/log file)."""
    print(f"  [rag] {msg}", flush=True)


# Cache the loaded index, and transparently reload it when ingestion rebuilds it.
_store = None
_store_mtime = None


def get_store():
    global _store, _store_mtime
    emb = config.INDEX_DIR / "embeddings.npy"
    if not emb.exists():
        _store, _store_mtime = None, None
        return None
    mtime = emb.stat().st_mtime
    if _store is None or mtime != _store_mtime:
        _store = VectorStore.load()
        _store_mtime = mtime
    return _store


class Handler(BaseHTTPRequestHandler):
    server_version = "rag-local"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, content_type, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _begin_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _event(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw or b"{}")
        return data if isinstance(data, dict) else {}

    # ---- routing ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (WEB_DIR / "index.html").read_text("utf-8"))
        elif path == "/api/status":
            store = get_store()
            h = health()
            by_base = {m.split(":", 1)[0]: m for m in h["models"]}  # base name -> full tag
            gen_full = by_base.get(config.GEN_MODEL.split(":", 1)[0])
            gen_det = h["model_details"].get(gen_full, {})
            # Cloud models (`*-cloud`) run on ollama.com, so they never appear in the local model
            # list — treat them as ready as long as Ollama itself is reachable (a missing sign-in
            # surfaces as an error on the actual /api/ask, not here).
            gen_cloud = config.GEN_MODEL.endswith("-cloud")
            self._send(200, "application/json", json.dumps({
                "index_exists": store is not None,
                "chunks": len(store) if store else 0,
                "gen_model": config.GEN_MODEL,
                "gen_cloud": gen_cloud,
                "embed_model": config.EMBED_MODEL,
                "ollama": h["ok"],
                "ollama_version": h["version"],
                "gen_ready": gen_cloud or config.GEN_MODEL.split(":", 1)[0] in by_base,
                "embed_ready": config.EMBED_MODEL.split(":", 1)[0] in by_base,
                "gen_version": gen_full,
                "gen_params": gen_det.get("parameter_size"),
                "gen_quant": gen_det.get("quantization_level"),
                "gen_digest": gen_det.get("digest"),
                "data_dir": str(config.DATA_DIR),
            }))
        elif path == "/favicon.ico":
            self._send(204, "text/plain", b"")
        else:
            self._send(404, "text/plain", "Not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/ask":
            self._handle_ask()
        elif path == "/api/ingest":
            self._handle_ingest()
        elif path == "/api/analyze":
            self._handle_analyze()
        else:
            self._send(404, "text/plain", "Not found")

    def _handle_ask(self):
        try:
            body = self._read_json()
            question = (body.get("question") or "").strip()
            history = body.get("history") if isinstance(body.get("history"), list) else []
        except ValueError:
            self._send(400, "text/plain", "Bad request")
            return
        self._begin_stream()
        try:
            if not question:
                self._event({"type": "error", "message": "Empty question."})
                return
            store = get_store()
            if store is None or len(store) == 0:
                self._event({"type": "error", "message": "No index yet — ingest a path first."})
                return
            # Resolve a follow-up against the conversation before retrieving.
            standalone = condense_question(history, question)
            hits = retrieve(store, standalone)
            if not hits:
                # No passage cleared RAG_MIN_SCORE. Fall back to a computed collection overview
                # so corpus-level questions ("how many docs?", "what companies?") still work.
                self._event({"type": "sources", "sources": []})
                log('ask: "%s" -> no passage match; answering from corpus overview' % standalone[:80])
                for piece in chat_stream(overview_messages(store, standalone)):
                    self._event({"type": "token", "text": piece})
                self._event({"type": "done"})
                return
            rewrite = "" if standalone == question else ' (→ "%s")' % standalone[:60]
            log('ask: "%s"%s -> %s' % (question[:80], rewrite, ", ".join(
                f"{r['source']}#{r['chunk']}({s:.2f})" for r, s in hits)))
            self._event({"type": "sources", "sources": [
                {"n": i + 1, "source": rec["source"], "chunk": rec["chunk"],
                 "score": round(score, 3), "text": rec["text"]}
                for i, (rec, score) in enumerate(hits)
            ]})
            for piece in chat_stream(build_messages(question, hits, history)):
                self._event({"type": "token", "text": piece})
            self._event({"type": "done"})
        except OllamaError as e:
            self._event({"type": "error", "message": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away mid-stream

    def _handle_ingest(self):
        try:
            body = self._read_json()
            path = (body.get("path") or "").strip()
            replace = bool(body.get("replace"))
            dry_run = bool(body.get("dry_run"))
            cs, co = body.get("chunk_size"), body.get("chunk_overlap")
            chunk_size = int(cs) if cs not in (None, "") else None
            chunk_overlap = int(co) if co not in (None, "") else None
        except (ValueError, TypeError):
            self._send(400, "text/plain", "Bad request")
            return
        self._begin_stream()
        try:
            if not path:
                self._event({"type": "error", "message": "Enter a file or folder path."})
                return
            log(f"ingest: path={path!r} replace={replace} dry_run={dry_run} chunk={chunk_size}/{chunk_overlap}")

            def emit(e):
                self._event(e)
                if e.get("type") == "done":
                    log(f"ingest {'(dry-run) ' if e.get('dry_run') else ''}done: {e.get('documents')} file(s), "
                        f"{e.get('chunks')} chunk(s), total {e.get('total_chunks')}")

            ingest_paths([path], replace=replace, emit=emit,
                         chunk_size=chunk_size, chunk_overlap=chunk_overlap, dry_run=dry_run)
        except (OllamaError, RuntimeError) as e:
            self._event({"type": "error", "message": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_analyze(self):
        """Profile the docs at a path and recommend a chunk size (no embedding). Returns JSON."""
        try:
            path = (self._read_json().get("path") or "").strip()
        except ValueError:
            self._send(400, "text/plain", "Bad request")
            return
        if not path:
            self._send(200, "application/json", json.dumps({"error": "Enter a file or folder path."}))
            return
        try:
            result = analyze_paths([path])
        except Exception as e:  # noqa: BLE001 - report any analysis failure to the UI
            self._send(200, "application/json", json.dumps({"error": str(e)}))
            return
        rec = result.get("recommended")
        log(f"analyze: path={path!r} -> {result['documents']} doc(s)"
            + (f", recommend {rec['chunk_size']}/{rec['chunk_overlap']}" if rec else ""))
        self._send(200, "application/json", json.dumps(result))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else int(os.environ.get("RAG_PORT", "8000"))
    if not (WEB_DIR / "index.html").exists():
        print(f"Missing {WEB_DIR / 'index.html'}", file=sys.stderr)
        return 1
    store = get_store()
    status = f"{len(store)} chunks indexed" if store else "no index yet — ingest a path in the UI"
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"rag-local web UI  →  http://127.0.0.1:{port}   ({status})")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
