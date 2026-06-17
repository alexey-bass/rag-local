"""Thin client over the local Ollama HTTP API (no third-party HTTP deps).

Exposes two things the RAG pipeline needs:
  - embed(texts)        -> numpy array of embedding vectors
  - chat_stream(...)    -> generator yielding answer text as it's produced
"""
import json
import urllib.error
import urllib.request

import numpy as np

from . import config


class OllamaError(RuntimeError):
    """Raised with a human-friendly message when Ollama is unreachable or a model is missing."""


def _open(path, payload):
    url = config.OLLAMA_HOST.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        hint = ""
        if "not found" in detail or "pulling" in detail:
            hint = f"\n  Try: ollama pull {payload.get('model', '<model>')}"
        raise OllamaError(f"Ollama returned HTTP {e.code}: {detail}{hint}") from e
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {config.OLLAMA_HOST}.\n"
            f"  Is it running? Start it with:  ollama serve\n  ({e.reason})"
        ) from e


def embed(texts, prefix=""):
    """Embed a string or list of strings. Returns an (N, dim) float32 numpy array.

    `prefix` is prepended to each input before embedding (e.g. nomic's
    "search_document: " / "search_query: " task instructions). It affects only what
    is sent to the model — callers keep their original, un-prefixed text.
    """
    if isinstance(texts, str):
        texts = [texts]
    if prefix:
        texts = [prefix + t for t in texts]
    resp = _open("/api/embed", {"model": config.EMBED_MODEL, "input": texts})
    body = json.loads(resp.read())
    if "embeddings" not in body:
        raise OllamaError(f"Unexpected embed response: {body}")
    return np.array(body["embeddings"], dtype=np.float32)


def health(timeout=2.0):
    """Probe the Ollama backend for the UI status indicator. Never raises.

    Returns a dict: {ok, version, models: [names], model_details: {name: {...}}}.
    `ok=False` simply means the server isn't reachable right now.
    """
    base = config.OLLAMA_HOST.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read())
    except Exception:  # noqa: BLE001 - unreachable is a normal, expected state
        return {"ok": False, "version": None, "models": [], "model_details": {}}

    details = {}
    for m in tags.get("models", []):
        d = m.get("details") or {}
        details[m.get("name", "")] = {
            "parameter_size": d.get("parameter_size"),
            "quantization_level": d.get("quantization_level"),
            "family": d.get("family"),
            "digest": (m.get("digest") or "").replace("sha256:", "")[:12],
        }

    version = None
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=timeout) as resp:
            version = json.loads(resp.read()).get("version")
    except Exception:  # noqa: BLE001 - version is best-effort
        version = None

    return {"ok": True, "version": version, "models": list(details.keys()), "model_details": details}


def chat_stream(system, user):
    """Stream a chat completion. Yields text chunks as the model produces them."""
    payload = {
        "model": config.GEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": config.GEN_TEMPERATURE},
    }
    resp = _open("/api/chat", payload)
    for line in resp:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("error"):
            raise OllamaError(obj["error"])
        chunk = obj.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if obj.get("done"):
            break
