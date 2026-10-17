from __future__ import annotations

"""
Semantic search (Query-time Step 5): embeddings-based retrieval.

This module integrates with the existing project pipeline:

Index-time
---------
- Step A: scan PDFs -> populate `documents`
- Step B: extract text -> populate `pages` and `lines` (and FTS5 indices)

Semantic (this step)
--------------------
- Build semantic "chunks" from the already-indexed `lines` table (quote-friendly windows)
- Compute embeddings for chunks using a pluggable embedder backend (see `src.index.embeddings`)
- Store:
    - chunks in SQLite table `chunks`
    - embeddings in SQLite table `chunk_embeddings` (supports multiple models)

Query-time
----------
- Embed user query
- Retrieve top-K closest chunks (MVP: brute-force cosine scan in SQLite)
- Return hits with citations (rel_path + page_no + line range) and chunk text

Design goals
------------
- Clean structure and composability with previous query-time steps:
    (1) query formation
    (2) typos
    (3) synonyms/morphology
    (4) BM25/FTS
    (5) embeddings (this module)
    (6) reranking (future)
- Match the shared SQLite schema defined in `src/index/schema.sql`:
    - `chunks(page_id, start_line_no, end_line_no, text, text_len, ...)`
    - `chunk_embeddings(chunk_id, model, dim, vector, ...)`
- Keep dependencies minimal:
    - no hard dependency on numpy/faiss/hnsw for MVP
    - vector index acceleration can be added later (see `src.index.vector_index`)

Important notes
---------------
- Embedding model availability is environment-dependent:
    - If you use Ollama embeddings: you need Ollama running and an embedding model installed.
    - For offline/dev smoke-tests: use the hash embedder from `src.index.embeddings`.
- This module does NOT modify PDF files and does NOT require re-running Step B unless
  the corpus changes.
"""

import math
import sqlite3
import time
from array import array
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from . import text_clean
from .embeddings import Embedder, EmbedderInfo, EmbeddingsError, stable_model_label

# -----------------------------
# Models / options
# -----------------------------


@dataclass(frozen=True)
class SemanticBuildOptions:
    # Chunking parameters (line windows per page)
    window_lines: int = 5
    stride: int = 2

    # Filtering
    min_line_len: int = 2
    min_chunk_chars: int = 80

    # Embedding batching
    batch_size: int = 32

    # Storage behavior
    clear_existing: bool = False
    incremental: bool = True  # skip embedding if already present for (chunk_id, model)

    # If True: only create chunks for pages that are OCR (pages.is_ocr=1)
    ocr_only: bool = False


@dataclass(frozen=True)
class SemanticBuildStats:
    chunks_seen: int
    chunks_created: int
    chunks_embedded: int
    model: str
    dim: int
    seconds: float


@dataclass(frozen=True)
class SemanticQueryOptions:
    top_k: int = 10
    min_score: float = -1.0  # cosine similarity threshold

    # Optional filter
    rel_path: Optional[str] = None

    # If True: return chunk text (useful for CLI); otherwise return empty text
    include_text: bool = True


@dataclass(frozen=True)
class SemanticHit:
    chunk_id: int
    score: float

    rel_path: str
    page_no: int
    start_line_no: int
    end_line_no: int

    text: str


class SemanticError(RuntimeError):
    """User-facing error for semantic indexing/search failures."""


# -----------------------------
# Schema helpers (match schema.sql)
# -----------------------------


def ensure_semantic_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure semantic tables exist.

    Note: The project already defines these tables in `src/index/schema.sql`.
    This function is kept for robustness (e.g., when using a DB created before
    this step existed). It uses CREATE IF NOT EXISTS, so it's safe to call.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id      INTEGER PRIMARY KEY,
          page_id       INTEGER NOT NULL,
          start_line_no INTEGER,
          end_line_no   INTEGER,
          text          TEXT NOT NULL,
          text_len      INTEGER NOT NULL,
          created_at    INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_page_lines ON chunks(page_id, start_line_no, end_line_no)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
          embed_id    INTEGER PRIMARY KEY,
          chunk_id    INTEGER NOT NULL,
          model       TEXT NOT NULL,
          dim         INTEGER NOT NULL,
          vector      BLOB NOT NULL,
          created_at  INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
          FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
          UNIQUE(chunk_id, model)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk ON chunk_embeddings(chunk_id)"
    )


def clear_semantic_data(conn: sqlite3.Connection) -> None:
    """
    Remove all semantic artifacts (chunks + embeddings).

    Deleting chunks will cascade-delete chunk_embeddings via FK.
    """
    ensure_semantic_schema(conn)
    conn.execute("DELETE FROM chunks")


# -----------------------------
# Chunking from `lines`
# -----------------------------


def _iter_pages(
    conn: sqlite3.Connection, *, ocr_only: bool
) -> Iterator[Tuple[int, int, str, int, int]]:
    """
    Yield pages as tuples:
      (page_id, doc_id, rel_path, page_no, is_ocr)
    """
    where = "WHERE p.is_ocr = 1" if ocr_only else ""
    rows = conn.execute(
        f"""
        SELECT p.page_id, d.doc_id, d.rel_path, p.page_no, p.is_ocr
        FROM pages p
        JOIN documents d ON d.doc_id = p.doc_id
        {where}
        ORDER BY d.rel_path, p.page_no
        """
    )
    for r in rows:
        yield int(r[0]), int(r[1]), str(r[2]), int(r[3]), int(r[4])


def _load_lines_for_page(
    conn: sqlite3.Connection, page_id: int, *, min_line_len: int
) -> List[Tuple[int, str]]:
    rows = conn.execute(
        """
        SELECT line_no, text
        FROM lines
        WHERE page_id = ?
        ORDER BY line_no
        """,
        (int(page_id),),
    ).fetchall()

    out: List[Tuple[int, str]] = []
    for ln, txt in rows:
        s = (txt or "").strip()
        if len(s) < int(min_line_len):
            continue
        out.append((int(ln), s))
    return out


def _iter_line_windows(
    lines: Sequence[Tuple[int, str]],
    *,
    window_lines: int,
    stride: int,
) -> Iterator[Tuple[int, int, str]]:
    if int(window_lines) <= 0:
        raise ValueError("window_lines must be > 0")
    if int(stride) <= 0:
        raise ValueError("stride must be > 0")
    if not lines:
        return

    n = len(lines)
    i = 0
    while i < n:
        chunk = lines[i : i + int(window_lines)]
        if not chunk:
            break
        start_ln = chunk[0][0]
        end_ln = chunk[-1][0]
        text = "\n".join(t for _ln, t in chunk).strip()
        yield int(start_ln), int(end_ln), str(text)

        if i + int(window_lines) >= n:
            break
        i += int(stride)


def _get_existing_chunk_id(
    conn: sqlite3.Connection,
    *,
    page_id: int,
    start_line_no: int,
    end_line_no: int,
) -> Optional[int]:
    row = conn.execute(
        """
        SELECT chunk_id
        FROM chunks
        WHERE page_id = ? AND start_line_no = ? AND end_line_no = ?
        LIMIT 1
        """,
        (int(page_id), int(start_line_no), int(end_line_no)),
    ).fetchone()
    return int(row[0]) if row else None


def _insert_chunk(
    conn: sqlite3.Connection,
    *,
    page_id: int,
    start_line_no: int,
    end_line_no: int,
    text: str,
) -> int:
    t = text or ""
    cur = conn.execute(
        """
        INSERT INTO chunks(page_id, start_line_no, end_line_no, text, text_len)
        VALUES(?, ?, ?, ?, ?)
        """,
        (page_id, start_line_no, end_line_no, t, len(t)),
    )
    rowid = cur.lastrowid
    if rowid is None:
        raise SemanticError("Failed to insert chunk: lastrowid is None")
    return int(rowid)


# -----------------------------
# Embedding storage helpers
# -----------------------------


def _embedding_exists(conn: sqlite3.Connection, *, chunk_id: int, model: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM chunk_embeddings
        WHERE chunk_id = ? AND model = ?
        LIMIT 1
        """,
        (int(chunk_id), str(model)),
    ).fetchone()
    return row is not None


def _vector_to_f32_blob(vec: Sequence[float]) -> Tuple[int, bytes]:
    arr = array("f", [float(x) for x in vec])
    if len(arr) <= 0:
        raise ValueError("empty embedding vector")
    return int(len(arr)), arr.tobytes()


def _upsert_embedding(
    conn: sqlite3.Connection,
    *,
    chunk_id: int,
    model: str,
    vec: Sequence[float],
) -> int:
    dim, blob = _vector_to_f32_blob(vec)
    conn.execute(
        """
        INSERT INTO chunk_embeddings(chunk_id, model, dim, vector)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(chunk_id, model) DO UPDATE SET
          dim=excluded.dim,
          vector=excluded.vector
        """,
        (int(chunk_id), str(model), int(dim), sqlite3.Binary(blob)),
    )
    return int(dim)


# -----------------------------
# Build semantic index (chunks + embeddings)
# -----------------------------


def build_semantic_index(
    conn: sqlite3.Connection,
    embedder: Embedder,
    *,
    options: Optional[SemanticBuildOptions] = None,
) -> SemanticBuildStats:
    """
    Build semantic chunks and embeddings for the given embedder.

    This does NOT require re-reading PDFs; it uses the indexed `lines` table.

    Behavior:
    - If clear_existing=True: deletes all chunks (and embeddings) and rebuilds.
    - Else: reuses existing chunks for same (page_id, start_line_no, end_line_no).
    - Embeddings are stored per-model via UNIQUE(chunk_id, model).
    """
    opts = options or SemanticBuildOptions()
    ensure_semantic_schema(conn)

    if opts.clear_existing:
        clear_semantic_data(conn)
        conn.commit()

    info: EmbedderInfo = embedder.info()
    model = stable_model_label(info)

    t0 = time.time()
    chunks_seen = 0
    chunks_created = 0
    chunks_embedded = 0
    dim_used = 0

    batch_texts: List[str] = []
    batch_chunk_ids: List[int] = []

    def _flush_batch() -> None:
        nonlocal chunks_embedded, dim_used
        if not batch_texts:
            return
        vecs = embedder.embed_texts(batch_texts)
        if len(vecs) != len(batch_texts):
            raise SemanticError(
                f"Embedder returned {len(vecs)} vectors for {len(batch_texts)} texts"
            )

        for cid, v in zip(batch_chunk_ids, vecs, strict=True):
            d = _upsert_embedding(conn, chunk_id=int(cid), model=model, vec=v)
            if dim_used == 0:
                dim_used = int(d)
            elif dim_used != int(d):
                raise SemanticError(
                    f"Inconsistent embedding dim: expected {dim_used}, got {d}"
                )
            chunks_embedded += 1

        batch_texts.clear()
        batch_chunk_ids.clear()

    for page_id, _doc_id, _rel_path, _page_no, _is_ocr in _iter_pages(
        conn, ocr_only=bool(opts.ocr_only)
    ):
        lines = _load_lines_for_page(conn, page_id, min_line_len=int(opts.min_line_len))
        if not lines:
            continue

        for start_ln, end_ln, chunk_text in _iter_line_windows(
            lines, window_lines=int(opts.window_lines), stride=int(opts.stride)
        ):
            # Normalize chunk text similarly to index-time
            chunk_text = text_clean.normalize_unicode(chunk_text)
            chunk_text = text_clean.normalize_newlines(chunk_text).strip()

            if len(chunk_text) < int(opts.min_chunk_chars):
                continue

            chunks_seen += 1

            existing = _get_existing_chunk_id(
                conn,
                page_id=int(page_id),
                start_line_no=int(start_ln),
                end_line_no=int(end_ln),
            )
            if existing is None:
                chunk_id = _insert_chunk(
                    conn,
                    page_id=int(page_id),
                    start_line_no=int(start_ln),
                    end_line_no=int(end_ln),
                    text=str(chunk_text),
                )
                chunks_created += 1
            else:
                chunk_id = int(existing)

            if bool(opts.incremental) and _embedding_exists(
                conn, chunk_id=int(chunk_id), model=model
            ):
                continue

            batch_texts.append(chunk_text)
            batch_chunk_ids.append(int(chunk_id))

            if len(batch_texts) >= int(opts.batch_size):
                _flush_batch()
                conn.commit()

    _flush_batch()
    conn.commit()

    return SemanticBuildStats(
        chunks_seen=int(chunks_seen),
        chunks_created=int(chunks_created),
        chunks_embedded=int(chunks_embedded),
        model=model,
        dim=int(dim_used or info.dim or 0),
        seconds=float(time.time() - t0),
    )


# -----------------------------
# Semantic search (MVP: brute-force cosine scan)
# -----------------------------


def _decode_f32(blob: bytes) -> List[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a)


def _l2_norm(vec: Sequence[float]) -> float:
    s2 = 0.0
    for x in vec:
        s2 += float(x) * float(x)
    return math.sqrt(s2) if s2 > 0.0 else 0.0


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b, strict=False):
        s += float(x) * float(y)
    return s


def semantic_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    user_query: str,
    *,
    options: Optional[SemanticQueryOptions] = None,
) -> List[SemanticHit]:
    """
    Return top-K semantic hits by cosine similarity.

    Requirements:
    - You must run build_semantic_index() for the same embedding model (embedder.info().model)
      at least once, so `chunk_embeddings` contains vectors for that model.

    Notes:
    - This MVP uses a brute-force scan; for large corpora, add an ANN index later.
    """
    ensure_semantic_schema(conn)
    opts = options or SemanticQueryOptions()

    q = (user_query or "").strip()
    q = text_clean.normalize_unicode(q)
    q = q.replace("\n", " ").replace("\r", " ")
    q = " ".join(q.split()).strip()
    if not q:
        return []

    info = embedder.info()
    model = stable_model_label(info)

    try:
        q_vec = embedder.embed_query(q)
    except EmbeddingsError as e:
        raise SemanticError(f"Failed to embed query with model={model!r}: {e}") from e

    qn = _l2_norm(q_vec)

    # Fetch chunk vectors + citations. Join to pages/documents to get rel_path/page_no.
    where = "WHERE e.model = ?"
    params: List[object] = [model]
    if opts.rel_path:
        where += " AND d.rel_path = ?"
        params.append(str(opts.rel_path))

    rows = conn.execute(
        f"""
        SELECT
          c.chunk_id,
          d.rel_path,
          p.page_no,
          COALESCE(c.start_line_no, 0) AS start_line_no,
          COALESCE(c.end_line_no, 0) AS end_line_no,
          c.text,
          e.dim,
          e.vector
        FROM chunk_embeddings e
        JOIN chunks c   ON c.chunk_id = e.chunk_id
        JOIN pages p    ON p.page_id  = c.page_id
        JOIN documents d ON d.doc_id  = p.doc_id
        {where}
        """,
        tuple(params),
    ).fetchall()

    scored: List[Tuple[float, SemanticHit]] = []

    # If embedder outputs normalized vectors, cosine sim can be computed as dot product.
    assume_normalized = bool(info.normalized)

    for r in rows:
        chunk_id = int(r[0])
        rel_path = str(r[1])
        page_no = int(r[2])
        start_ln = int(r[3])
        end_ln = int(r[4])
        text = str(r[5]) if bool(opts.include_text) else ""
        dim = int(r[6])
        blob = r[7]

        if not isinstance(blob, (bytes, bytearray)):
            continue
        if dim != len(q_vec):
            # likely different embedding model; skip
            continue

        v = _decode_f32(bytes(blob))
        if len(v) != len(q_vec):
            continue

        if assume_normalized:
            score = _dot(q_vec, v)
        else:
            vn = _l2_norm(v)
            if qn <= 0.0 or vn <= 0.0:
                score = 0.0
            else:
                score = _dot(q_vec, v) / (qn * vn)

        if score < float(opts.min_score):
            continue

        hit = SemanticHit(
            chunk_id=int(chunk_id),
            score=float(score),
            rel_path=rel_path,
            page_no=int(page_no),
            start_line_no=int(start_ln),
            end_line_no=int(end_ln),
            text=text,
        )
        scored.append((float(score), hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(0, int(opts.top_k))]
    return [h for _s, h in top]


__all__ = [
    "SemanticBuildOptions",
    "SemanticBuildStats",
    "SemanticQueryOptions",
    "SemanticHit",
    "SemanticError",
    "ensure_semantic_schema",
    "clear_semantic_data",
    "build_semantic_index",
    "semantic_search",
]
