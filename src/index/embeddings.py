from __future__ import annotations

"""
Embedding backends for Query-time Step (5): semantic search.

This module is designed to be:
- cleanly layered (pure query-time utility; no DB knowledge here)
- dependency-light (stdlib-only by default)
- extensible (add new backends without changing call sites)
- safe for offline/dev (deterministic hash fallback embedder)

Backends included
-----------------
1) HashEmbedder (always available)
   - Deterministic pseudo-embeddings from a stable hash.
   - Useful for testing the end-to-end vector pipeline without any external model.

2) OllamaEmbedder (optional; requires a running Ollama server)
   - Uses HTTP calls to Ollama embeddings endpoint.
   - No extra Python dependencies required.

3) SentenceTransformerEmbedder (optional; requires `sentence_transformers`)
   - Uses local sentence-transformers models (may require torch/numpy).

Typical usage
-------------
You will typically build an embedder once, then call embed_texts() for:
- indexing: embed many chunks
- query-time: embed the user query (embed_query())

Example:
    embedder = create_embedder(
        preferred=["ollama", "sentence_transformers", "hash"],
        ollama_model="nomic-embed-text",
        hash_dim=384,
    )
    q_vec = embedder.embed_query("напряжение reference voltage")
    # then search your vector index with q_vec

Notes
-----
- This module intentionally does NOT implement a vector index (FAISS/HNSW/etc.)
  and does NOT implement chunking. Those belong in separate modules/steps.
- All vectors returned by this module are float lists. You can store them in
  numpy arrays or serialize as needed.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Iterable, List, Literal, Optional, Sequence

# Optional: align query normalization with index-time rules if available.
try:  # pragma: no cover
    from . import text_clean as _text_clean
except Exception:  # pragma: no cover
    _text_clean = None  # type: ignore[assignment]


BackendName = Literal["hash", "ollama", "sentence_transformers"]


class EmbeddingsError(RuntimeError):
    """User-facing error for embedding-related failures."""


@dataclass(frozen=True)
class EmbedderInfo:
    backend: BackendName
    model: str
    dim: int
    normalized: bool


def stable_model_label(info: EmbedderInfo) -> str:
    """
    Build a stable label for storing/retrieving embeddings across the project.

    This label is what you should store in SQLite (e.g. chunk_embeddings.model) and
    later use again at query-time to select the correct vectors.

    Rules:
      - For hash backend, `info.model` already contains a "hash:..." prefix (and includes dim),
        so we keep it as-is.
      - For all other backends, we prefix with "<backend>:".
    """
    backend = str(info.backend)
    model = str(info.model)

    if backend == "hash" and model.startswith("hash:"):
        return model
    return f"{backend}:{model}"


def _now_s() -> int:
    return int(time.time())


def _normalize_text_for_embedding(text: str) -> str:
    """
    Minimal normalization for embedding inputs.

    We keep it conservative:
    - unicode normalization (if index text_clean is present)
    - strip and collapse whitespace
    """
    s = (text or "").strip()
    if _text_clean is not None:
        try:
            s = _text_clean.normalize_unicode(s)
        except Exception:
            pass
    # collapse internal whitespace
    s = " ".join(s.split())
    return s


def _l2_normalize(vec: list[float]) -> list[float]:
    ss = 0.0
    for x in vec:
        ss += float(x) * float(x)
    if ss <= 0.0:
        return vec
    inv = 1.0 / math.sqrt(ss)
    return [float(x) * inv for x in vec]


def _blake2b_person(seed: bytes) -> bytes:
    """
    Build a valid blake2b `person` value (<= 16 bytes).
    If the seed is longer, we compress it deterministically to 16 bytes.

    Why:
      hashlib.blake2b(person=...) raises:
        ValueError: maximum person length is 16 bytes
    """
    b = seed or b""
    if len(b) <= 16:
        return b
    return blake2b(b, digest_size=16).digest()


class Embedder:
    """
    Base interface for embedding backends.
    """

    def info(self) -> EmbedderInfo:
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        vecs = self.embed_texts([text])
        if not vecs:
            raise EmbeddingsError("embed_query() returned no vectors")
        return vecs[0]


def stable_model_label_for_embedder(embedder: Embedder) -> str:
    """
    Convenience wrapper around stable_model_label(embedder.info()).
    """
    return stable_model_label(embedder.info())


# -----------------------------------------------------------------------------
# Hash fallback (deterministic pseudo-embeddings)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HashEmbedderConfig:
    dim: int = 384
    normalized: bool = True
    seed: str = "older.index.hash_embedder.v1"


class HashEmbedder(Embedder):
    """
    Deterministic, model-free embeddings derived from a hash.

    This is NOT semantic. It is only meant to:
    - keep the pipeline testable offline
    - provide stable dimensions for storage/indexing logic
    - support integration tests for vector retrieval without model dependencies
    """

    def __init__(self, config: Optional[HashEmbedderConfig] = None) -> None:
        cfg = config or HashEmbedderConfig()
        if int(cfg.dim) <= 0:
            raise ValueError("HashEmbedder dim must be > 0")
        self._cfg = cfg

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend="hash",
            model=f"hash:{int(self._cfg.dim)}:{self._cfg.seed}",
            dim=int(self._cfg.dim),
            normalized=bool(self._cfg.normalized),
        )

    def _hash_vec(self, text: str) -> list[float]:
        dim = int(self._cfg.dim)
        seed_raw = self._cfg.seed.encode("utf-8", errors="ignore")
        person = _blake2b_person(seed_raw)
        msg = _normalize_text_for_embedding(text).encode("utf-8", errors="ignore")

        out: list[float] = []
        counter = 0
        # Use BLAKE2b in counter mode to generate enough bytes for dim floats.
        # Each digest is 64 bytes; we convert each 4 bytes into one float.
        while len(out) < dim:
            h = blake2b(digest_size=64, person=person)
            h.update(counter.to_bytes(8, "little", signed=False))
            h.update(b"\0")
            h.update(msg)
            digest = h.digest()

            # Convert bytes -> unsigned ints -> floats in [-1, 1]
            for i in range(0, len(digest), 4):
                if len(out) >= dim:
                    break
                u = int.from_bytes(digest[i : i + 4], "little", signed=False)
                # Map 0..2^32-1 to [-1, 1]
                x = (u / 4294967295.0) * 2.0 - 1.0
                out.append(float(x))

            counter += 1

        if self._cfg.normalized:
            out = _l2_normalize(out)
        return out

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if texts is None:
            raise ValueError("texts must not be None")
        return [self._hash_vec(t) for t in texts]


# -----------------------------------------------------------------------------
# Ollama embeddings backend (optional)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaEmbedderConfig:
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout_s: int = 60
    normalized: bool = True

    # Newer Ollama supports /api/embed with batch input; older supports /api/embeddings (single).
    prefer_batch_endpoint: bool = True


class OllamaEmbedder(Embedder):
    """
    Embedder that calls Ollama over HTTP.

    Endpoints (as seen across versions):
      - POST /api/embed       { "model": "...", "input": ["...", "..."] } -> { "embeddings": [[...], ...] }
      - POST /api/embeddings  { "model": "...", "prompt": "..." } -> { "embedding": [...] }

    We support both; we try batch first (if configured), then fall back to single.
    """

    def __init__(self, config: Optional[OllamaEmbedderConfig] = None) -> None:
        self._cfg = config or OllamaEmbedderConfig()

        if not self._cfg.model:
            raise ValueError("Ollama model must be non-empty")
        if not self._cfg.base_url:
            raise ValueError("Ollama base_url must be non-empty")
        if int(self._cfg.timeout_s) <= 0:
            raise ValueError("Ollama timeout_s must be > 0")

        self._cached_dim: Optional[int] = None

    def info(self) -> EmbedderInfo:
        # Dim is unknown until first call; report 0 if not yet known.
        return EmbedderInfo(
            backend="ollama",
            model=str(self._cfg.model),
            dim=int(self._cached_dim or 0),
            normalized=bool(self._cfg.normalized),
        )

    def _http_post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._cfg.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=int(self._cfg.timeout_s)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raw = (
                e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            )
            raise EmbeddingsError(f"Ollama HTTP error {e.code} for {url}: {raw}") from e
        except Exception as e:
            raise EmbeddingsError(f"Failed to call Ollama at {url}: {e}") from e

        try:
            return json.loads(body)
        except Exception as e:
            raise EmbeddingsError(
                f"Invalid JSON from Ollama at {url}: {body[:2000]!r}"
            ) from e

    def _embed_batch(self, texts: Sequence[str]) -> Optional[list[list[float]]]:
        payload = {"model": self._cfg.model, "input": list(texts)}
        try:
            data = self._http_post_json("/api/embed", payload)
        except EmbeddingsError:
            return None

        embs = data.get("embeddings")
        if not isinstance(embs, list):
            return None

        out: list[list[float]] = []
        for v in embs:
            if not isinstance(v, list):
                return None
            out.append([float(x) for x in v])

        return out

    def _embed_single(self, text: str) -> list[float]:
        payload = {"model": self._cfg.model, "prompt": text}
        data = self._http_post_json("/api/embeddings", payload)
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise EmbeddingsError(
                "Ollama /api/embeddings response missing 'embedding' list"
            )
        return [float(x) for x in emb]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if texts is None:
            raise ValueError("texts must not be None")

        norm_texts = [_normalize_text_for_embedding(t) for t in texts]

        vecs: Optional[list[list[float]]] = None
        if self._cfg.prefer_batch_endpoint and len(norm_texts) > 1:
            vecs = self._embed_batch(norm_texts)

        if vecs is None:
            # Fall back to single-call endpoint (works for all versions).
            vecs = [self._embed_single(t) for t in norm_texts]

        if not vecs:
            raise EmbeddingsError("Ollama returned no embeddings")

        if self._cfg.normalized:
            vecs = [_l2_normalize(v) for v in vecs]

        # Cache dim after success
        if self._cached_dim is None and vecs and vecs[0]:
            self._cached_dim = int(len(vecs[0]))

        return vecs


# -----------------------------------------------------------------------------
# sentence-transformers backend (optional)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceTransformerEmbedderConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalized: bool = True
    device: Optional[str] = None  # e.g. "cpu", "cuda"
    batch_size: int = 32


class SentenceTransformerEmbedder(Embedder):
    """
    Local embeddings using sentence-transformers.

    Requires:
      pip install sentence-transformers

    Notes:
    - This backend may pull model weights (internet) unless already cached locally.
    - For minimal environments, prefer Ollama or HashEmbedder.
    """

    def __init__(
        self, config: Optional[SentenceTransformerEmbedderConfig] = None
    ) -> None:
        cfg = config or SentenceTransformerEmbedderConfig()
        self._cfg = cfg

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise EmbeddingsError(
                "sentence-transformers is not installed. Install with: pip install sentence-transformers"
            ) from e

        kwargs: dict[str, Any] = {}
        if cfg.device:
            kwargs["device"] = cfg.device

        self._model = SentenceTransformer(cfg.model, **kwargs)
        self._cached_dim: Optional[int] = None

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend="sentence_transformers",
            model=str(self._cfg.model),
            dim=int(self._cached_dim or 0),
            normalized=bool(self._cfg.normalized),
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if texts is None:
            raise ValueError("texts must not be None")

        norm_texts = [_normalize_text_for_embedding(t) for t in texts]

        # sentence-transformers returns numpy array by default; convert to Python lists.
        vecs_any = self._model.encode(
            norm_texts,
            batch_size=int(self._cfg.batch_size),
            normalize_embeddings=bool(self._cfg.normalized),
            show_progress_bar=False,
        )

        # vecs_any may be numpy array or list; handle both without importing numpy explicitly.
        out: list[list[float]] = []
        try:
            for row in vecs_any:
                out.append([float(x) for x in row])
        except TypeError:
            # If encode returned a single vector for a single input
            out = [[float(x) for x in vecs_any]]

        if not out:
            raise EmbeddingsError("sentence-transformers returned no embeddings")

        if self._cached_dim is None and out and out[0]:
            self._cached_dim = int(len(out[0]))

        return out


# -----------------------------------------------------------------------------
# Factory / availability
# -----------------------------------------------------------------------------


def available_backends() -> list[BackendName]:
    out: list[BackendName] = ["hash"]

    # sentence-transformers availability (do not import the package here)
    # Using find_spec avoids optional import warnings/errors in type checkers.
    try:  # pragma: no cover
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is not None:
            out.append("sentence_transformers")
    except Exception:
        # If importlib is unavailable or spec lookup fails, treat as not available.
        pass

    # ollama doesn't require an import; runtime availability is checked via HTTP.
    out.append("ollama")
    return out


def create_embedder(
    *,
    preferred: Sequence[BackendName] = ("ollama", "sentence_transformers", "hash"),
    # Ollama
    ollama_model: str = "nomic-embed-text",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout_s: int = 60,
    # sentence-transformers
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    st_device: Optional[str] = None,
    st_batch_size: int = 32,
    # hash fallback
    hash_dim: int = 384,
    hash_seed: str = "older.index.hash_embedder.v1",
    # common
    normalized: bool = True,
) -> Embedder:
    """
    Create an embedder from a preference-ordered list.

    The factory chooses the first backend that is usable in the current environment:
      - ollama: always constructible, but will fail at runtime if server/model missing
      - sentence_transformers: requires package installed (and model availability)
      - hash: always available
    """
    if not preferred:
        preferred = ("hash",)

    last_error: Optional[Exception] = None

    for name in preferred:
        if name == "ollama":
            try:
                return OllamaEmbedder(
                    OllamaEmbedderConfig(
                        model=str(ollama_model),
                        base_url=str(ollama_base_url),
                        timeout_s=int(ollama_timeout_s),
                        normalized=bool(normalized),
                        prefer_batch_endpoint=True,
                    )
                )
            except Exception as e:
                last_error = e
                continue

        if name == "sentence_transformers":
            try:
                return SentenceTransformerEmbedder(
                    SentenceTransformerEmbedderConfig(
                        model=str(st_model),
                        normalized=bool(normalized),
                        device=st_device,
                        batch_size=int(st_batch_size),
                    )
                )
            except Exception as e:
                last_error = e
                continue

        if name == "hash":
            return HashEmbedder(
                HashEmbedderConfig(
                    dim=int(hash_dim),
                    normalized=bool(normalized),
                    seed=str(hash_seed),
                )
            )

    if last_error is not None:
        raise EmbeddingsError(
            f"Failed to create embedder (last error: {last_error})"
        ) from last_error

    raise EmbeddingsError("Failed to create embedder: no backends available")
