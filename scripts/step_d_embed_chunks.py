from __future__ import annotations

"""
Step D (Index-time): build semantic chunks + embeddings into SQLite.

This script implements Query-time Step (5) prerequisites: embeddings-based semantic search.

It builds on the existing pipeline:
- Step A: scripts/step_a_scan_corpus.py   -> documents registry
- Step B: scripts/step_b_index_pages.py   -> pages + lines + FTS5 (lexical index)
- Step D: (this script)                  -> chunks + chunk_embeddings (semantic index artifacts)

What it does
------------
1) Reads indexed `pages` + `lines` from SQLite.
2) Creates quote-friendly "chunks" per page as windows of consecutive lines:
     window_lines=5, stride=2  (configurable)
3) Stores chunks in SQLite table `chunks`:
     chunks(page_id, start_line_no, end_line_no, text, text_len, ...)
4) Computes embeddings for each chunk and stores them in `chunk_embeddings`:
     chunk_embeddings(chunk_id, model, dim, vector BLOB)

Key design goals
----------------
- Clean structure: keeps semantic indexing separate from Step B.
- Safe & explainable: chunks are still tied to page + line range for citations.
- Extensible:
  - works with multiple embedding backends (Ollama / sentence-transformers / hash fallback)
  - supports multiple models via UNIQUE(chunk_id, model)
  - leaves room for a later ANN index (FAISS/HNSW) without changing DB tables.

Requirements
------------
- SQLite DB already created and populated by Step B.
- Embedding backend:
  - Recommended: Ollama running locally with an embedding model available (e.g. "nomic-embed-text")
  - Alternative: sentence-transformers (local)
  - Fallback: deterministic hash embedder (for offline smoke-testing; NOT semantic)

Usage
-----
From project root:
  python scripts/step_d_embed_chunks.py --embedder ollama --ollama-model nomic-embed-text

Common dev/test:
  python scripts/step_d_embed_chunks.py --embedder hash --hash-dim 256 --clear

Notes
-----
- By default, this script reuses existing chunks (same page + line range) to avoid duplicates.
- By default, it embeds only missing (chunk_id, model) rows (incremental mode).
- Use --clear to rebuild chunks (and embeddings for all models) from scratch.

"""

import argparse
import re
import sqlite3
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

# Allow running from any working directory: add project root (parent of scripts/) to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index import text_clean  # noqa: E402
from src.index.db import connect, init_db, transaction  # noqa: E402
from src.index.embeddings import (  # noqa: E402
    EmbeddingsError,
    create_embedder,
)


@dataclass
class RunStats:
    pages_seen: int = 0
    pages_skipped_no_lines: int = 0
    chunks_seen: int = 0
    chunks_inserted: int = 0
    chunks_reused: int = 0
    embeddings_added: int = 0
    embeddings_skipped_existing: int = 0
    embedding_dim: int = 0
    seconds: float = 0.0


def _model_label(embedder) -> str:
    """
    Create a stable model label for DB storage.

    Delegates to `embeddings.stable_model_label_for_embedder()` to ensure the same model
    naming is used across:
      - semantic indexing (Step D)
      - semantic query-time retrieval (Step C --semantic / hybrid)
    """
    from src.index.embeddings import stable_model_label_for_embedder

    return stable_model_label_for_embedder(embedder)


def _ensure_semantic_tables(conn: sqlite3.Connection) -> None:
    """
    Ensure semantic tables exist (schema.sql should already create them, but this
    keeps Step D robust even if DB was created externally).
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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
    return {str(r[1]) for r in rows}


def _has_unique_index_on_columns(
    conn: sqlite3.Connection, table: str, columns: Sequence[str]
) -> bool:
    """
    Best-effort check that `table` has a UNIQUE index exactly on `columns` (order-insensitive).

    SQLite will create an internal autoindex for UNIQUE constraints; this function
    considers those as well.
    """
    want = set(columns)
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    for r in rows:
        # PRAGMA index_list: seq, name, unique, origin, partial
        try:
            name = str(r[1])
            unique_raw = r[2]
            if unique_raw is None:
                continue
            unique = int(unique_raw)
        except Exception:
            continue
        if unique != 1:
            continue

        cols = conn.execute(f"PRAGMA index_info({name})").fetchall()
        idx_cols = {str(c[2]) for c in cols if c[2] is not None}  # seqno, cid, name
        if idx_cols == want:
            return True

    return False


def _validate_semantic_schema(conn: sqlite3.Connection) -> None:
    """
    Validate that semantic tables match what Step D expects.

    Why:
      - CREATE TABLE IF NOT EXISTS does not modify existing tables.
      - If a DB was created by an older/newer version of the project (or manually),
        silent schema mismatches can lead to confusing runtime behavior.

    This validation is intentionally minimal and focused on required columns and
    the UNIQUE(chunk_id, model) constraint.
    """
    # Validate chunks table
    required_chunks = {
        "chunk_id",
        "page_id",
        "start_line_no",
        "end_line_no",
        "text",
        "text_len",
        "created_at",
    }
    cols_chunks = _table_columns(conn, "chunks")
    missing_chunks = sorted(required_chunks - cols_chunks)
    if missing_chunks:
        raise RuntimeError(
            "Semantic schema mismatch: table `chunks` is missing required columns: "
            f"{missing_chunks}. "
            "Your DB was likely created with an incompatible schema. "
            "Fix: rebuild the DB (or use a fresh DB file), then re-run Step A + Step B, "
            "then re-run Step D."
        )

    # Validate chunk_embeddings table
    required_emb = {
        "embed_id",
        "chunk_id",
        "model",
        "dim",
        "vector",
        "created_at",
    }
    cols_emb = _table_columns(conn, "chunk_embeddings")
    missing_emb = sorted(required_emb - cols_emb)
    if missing_emb:
        raise RuntimeError(
            "Semantic schema mismatch: table `chunk_embeddings` is missing required columns: "
            f"{missing_emb}. "
            "Your DB was likely created with an incompatible schema. "
            "Fix: rebuild the DB (or use a fresh DB file), then re-run Step A + Step B, "
            "then re-run Step D."
        )

    # Validate uniqueness needed to support multiple models correctly.
    if not _has_unique_index_on_columns(
        conn, "chunk_embeddings", ["chunk_id", "model"]
    ):
        raise RuntimeError(
            "Semantic schema mismatch: table `chunk_embeddings` must have a UNIQUE constraint "
            "on (chunk_id, model) to support multiple embedding models safely. "
            "Fix: rebuild the DB using the current schema, then re-run Step A + Step B + Step D."
        )


def _clear_semantic(conn: sqlite3.Connection) -> None:
    _ensure_semantic_tables(conn)
    # Deleting chunks cascades to chunk_embeddings
    conn.execute("DELETE FROM chunks")


def _iter_pages(conn: sqlite3.Connection) -> Iterator[tuple[int, int, str]]:
    """
    Yield pages as tuples:
      (page_id, page_no, rel_path)
    """
    rows = conn.execute(
        """
        SELECT p.page_id, p.page_no, d.rel_path
        FROM pages p
        JOIN documents d ON d.doc_id = p.doc_id
        ORDER BY d.rel_path, p.page_no
        """
    )
    for r in rows:
        yield int(r[0]), int(r[1]), str(r[2])


def _fetch_lines(
    conn: sqlite3.Connection, page_id: int, *, min_len: int
) -> list[tuple[int, str]]:
    rows = conn.execute(
        """
        SELECT line_no, text
        FROM lines
        WHERE page_id=?
        ORDER BY line_no
        """,
        (int(page_id),),
    ).fetchall()

    out: list[tuple[int, str]] = []
    for ln, txt in rows:
        s = (txt or "").strip()
        if len(s) < int(min_len):
            continue
        out.append((int(ln), s))
    return out


def _iter_windows(
    lines: Sequence[tuple[int, str]],
    *,
    window_lines: int,
    stride: int,
) -> Iterator[tuple[int, int, str]]:
    """
    Convert [(line_no, text), ...] into windows:
      (start_line_no, end_line_no, joined_text)
    """
    if window_lines <= 0:
        raise ValueError("window_lines must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if not lines:
        return

    n = len(lines)
    i = 0
    while i < n:
        win = lines[i : i + window_lines]
        if not win:
            break

        start_ln = win[0][0]
        end_ln = win[-1][0]
        joined = "\n".join(t for _ln, t in win).strip()
        yield start_ln, end_ln, joined

        if i + window_lines >= n:
            break
        i += stride


def _load_existing_chunks_for_page(
    conn: sqlite3.Connection, page_id: int
) -> dict[tuple[int, int], int]:
    """
    Map (start_line_no, end_line_no) -> chunk_id for an existing page_id.
    Used to avoid duplicate chunk rows when re-running Step D.
    """
    rows = conn.execute(
        """
        SELECT chunk_id, start_line_no, end_line_no
        FROM chunks
        WHERE page_id=?
        """,
        (int(page_id),),
    ).fetchall()

    out: dict[tuple[int, int], int] = {}
    for cid, s, e in rows:
        if s is None or e is None:
            continue
        out[(int(s), int(e))] = int(cid)
    return out


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
    if cur.lastrowid is None:
        raise RuntimeError("Failed to insert chunk row: lastrowid is None")
    return int(cur.lastrowid)


def _chunk_embeddings_exist_set(
    conn: sqlite3.Connection, *, model: str, chunk_ids: Sequence[int]
) -> set[int]:
    """
    Return the subset of chunk_ids that already have an embedding for this model.
    """
    if not chunk_ids:
        return set()

    # SQLite has a variable limit; chunk in safe batches.
    existing: set[int] = set()
    BATCH = 900
    for i in range(0, len(chunk_ids), BATCH):
        part = chunk_ids[i : i + BATCH]
        placeholders = ",".join("?" for _ in part)
        rows = conn.execute(
            f"""
            SELECT chunk_id
            FROM chunk_embeddings
            WHERE model = ?
              AND chunk_id IN ({placeholders})
            """,
            (str(model), *[int(x) for x in part]),
        ).fetchall()
        for (cid,) in rows:
            existing.add(int(cid))
    return existing


def _upsert_embedding(
    conn: sqlite3.Connection,
    *,
    chunk_id: int,
    model: str,
    vec: Sequence[float],
) -> int:
    """
    Store float32 embedding as BLOB.
    Returns dim.
    """
    arr = array("f", [float(x) for x in vec])
    dim = len(arr)
    if dim <= 0:
        raise ValueError("empty embedding vector")

    conn.execute(
        """
        INSERT INTO chunk_embeddings(chunk_id, model, dim, vector)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(chunk_id, model) DO UPDATE SET
          dim=excluded.dim,
          vector=excluded.vector
        """,
        (int(chunk_id), str(model), int(dim), sqlite3.Binary(arr.tobytes())),
    )
    return int(dim)


def build_chunks_and_embeddings(
    conn: sqlite3.Connection,
    *,
    embedder,
    model: str,
    window_lines: int,
    stride: int,
    min_line_len: int,
    min_chunk_chars: int,
    batch_size: int,
    incremental: bool,
    reuse_chunks: bool,
    limit_pages: Optional[int],
    stats: RunStats,
) -> None:
    batch_texts: list[str] = []
    batch_chunk_ids: list[int] = []

    def _flush() -> None:
        if not batch_texts:
            return

        vecs = embedder.embed_texts(batch_texts)
        if len(vecs) != len(batch_texts):
            raise RuntimeError(
                f"Embedder returned {len(vecs)} vectors for {len(batch_texts)} texts"
            )

        for cid, v in zip(batch_chunk_ids, vecs, strict=True):
            dim = _upsert_embedding(conn, chunk_id=int(cid), model=str(model), vec=v)
            stats.embeddings_added += 1
            if stats.embedding_dim == 0:
                stats.embedding_dim = int(dim)
            elif stats.embedding_dim != int(dim):
                raise RuntimeError(
                    f"Inconsistent embedding dim: expected {stats.embedding_dim}, got {dim}"
                )

        batch_texts.clear()
        batch_chunk_ids.clear()

    for idx, (page_id, page_no, rel_path) in enumerate(_iter_pages(conn), start=1):
        if limit_pages is not None and idx > int(limit_pages):
            break

        stats.pages_seen += 1

        lines = _fetch_lines(conn, int(page_id), min_len=int(min_line_len))
        if not lines:
            stats.pages_skipped_no_lines += 1
            continue

        existing_map: dict[tuple[int, int], int] = {}
        if reuse_chunks:
            existing_map = _load_existing_chunks_for_page(conn, int(page_id))

        # Build candidate chunks (and reuse/insert)
        for start_ln, end_ln, raw_text in _iter_windows(
            lines, window_lines=int(window_lines), stride=int(stride)
        ):
            # normalize chunk text similarly to indexed text
            text = text_clean.normalize_unicode(raw_text)
            text = text_clean.normalize_newlines(text)
            text = text.strip()

            if len(text) < int(min_chunk_chars):
                continue

            stats.chunks_seen += 1

            key = (int(start_ln), int(end_ln))
            chunk_id: Optional[int] = None

            if reuse_chunks and key in existing_map:
                chunk_id = int(existing_map[key])
                stats.chunks_reused += 1
            else:
                chunk_id = _insert_chunk(
                    conn,
                    page_id=int(page_id),
                    start_line_no=int(start_ln),
                    end_line_no=int(end_ln),
                    text=text,
                )
                existing_map[key] = int(chunk_id)
                stats.chunks_inserted += 1

            # Incremental embedding: skip if exists
            if incremental:
                # we batch-check periodically; for now, add to batch and skip later if exists
                batch_texts.append(text)
                batch_chunk_ids.append(int(chunk_id))
            else:
                batch_texts.append(text)
                batch_chunk_ids.append(int(chunk_id))

            # flush when batch full (and commit outside)
            if len(batch_texts) >= int(batch_size):
                if incremental:
                    already = _chunk_embeddings_exist_set(
                        conn, model=str(model), chunk_ids=batch_chunk_ids
                    )
                    if already:
                        # filter out already-embedded chunks
                        keep_texts: list[str] = []
                        keep_ids: list[int] = []
                        for cid, txt in zip(batch_chunk_ids, batch_texts, strict=True):
                            if int(cid) in already:
                                stats.embeddings_skipped_existing += 1
                                continue
                            keep_ids.append(int(cid))
                            keep_texts.append(txt)
                        batch_chunk_ids[:] = keep_ids
                        batch_texts[:] = keep_texts

                _flush()
                conn.commit()

    # Final flush
    if incremental and batch_chunk_ids:
        already = _chunk_embeddings_exist_set(
            conn, model=str(model), chunk_ids=batch_chunk_ids
        )
        if already:
            keep_texts2: list[str] = []
            keep_ids2: list[int] = []
            for cid, txt in zip(batch_chunk_ids, batch_texts, strict=True):
                if int(cid) in already:
                    stats.embeddings_skipped_existing += 1
                    continue
                keep_ids2.append(int(cid))
                keep_texts2.append(txt)
            batch_chunk_ids[:] = keep_ids2
            batch_texts[:] = keep_texts2

    _flush()
    conn.commit()


def normalize_filename_for_embedding(rel_path: str) -> str:
    """
    Превращает относительный путь файла в читаемый заголовок для эмбеддинга.

    Примеры:
      'Лекция_4_ИНС.pdf'                      -> 'Лекция 4 ИНС'
      'subfolder/Backprop-Algorithm-v2.pdf'   -> 'Backprop Algorithm v2'
      '李强会见乌拉圭总统.docx'                -> '李强会见乌拉圭总统'
    """
    name = Path(rel_path).stem
    name = re.sub(r"[_\-\.\(\)\[\]\{\}]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_title_chunks_and_embeddings(
    conn: sqlite3.Connection,
    *,
    embedder,
    model: str,
    batch_size: int,
    stats: RunStats,
) -> None:
    """
    Для каждого документа создаёт один синтетический title-чанк:
      - page_no=0, start_line_no=0, end_line_no=0
      - text = normalize_filename_for_embedding(rel_path)
    И строит embedding для него (если ещё не построен для текущей модели).

    Это позволяет находить документы по запросам, совпадающим с названием
    файла, даже если внутри документа таких терминов нет.
    """
    # Берём только документы, у которых уже есть реальное содержимое (page_no>0).
    # Иначе мы плодим title-страницы для ошибочных файлов и сайдбар врёт.
    docs = conn.execute(
        """
        SELECT d.doc_id, d.rel_path
        FROM documents d
        WHERE EXISTS (
            SELECT 1 FROM pages p
            WHERE p.doc_id = d.doc_id AND p.page_no > 0
        )
        """
    ).fetchall()

    titles: list[tuple[int, str]] = []
    for doc_id, rel_path in docs:
        title = normalize_filename_for_embedding(str(rel_path))
        if not title or len(title) < 2:
            continue
        titles.append((int(doc_id), title))

    if not titles:
        print("Title pass: no documents to process.")
        return

    print(f"Title pass: ensuring title chunks for {len(titles)} documents...")

    pending_texts: list[str] = []
    pending_chunk_ids: list[int] = []
    reused = 0

    for doc_id, title in titles:
        # 1) Synthetic title page (page_no=0)
        row = conn.execute(
            "SELECT page_id FROM pages WHERE doc_id=? AND page_no=0",
            (doc_id,),
        ).fetchone()
        if row:
            page_id = int(row[0])
            # Обновляем текст, если файл переименовали.
            conn.execute(
                "UPDATE pages SET text=?, text_len=? WHERE page_id=?",
                (title, len(title), page_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO pages(doc_id, page_no, text, is_ocr, text_len) "
                "VALUES(?, 0, ?, 0, ?)",
                (doc_id, title, len(title)),
            )
            page_id = int(cur.lastrowid)

        # 2) Synthetic title chunk (start=0, end=0)
        row = conn.execute(
            "SELECT chunk_id, text FROM chunks "
            "WHERE page_id=? AND start_line_no=0 AND end_line_no=0",
            (page_id,),
        ).fetchone()
        if row:
            chunk_id = int(row[0])
            if str(row[1] or "") != title:
                # Файл переименовали — обновляем чанк и инвалидируем embedding.
                conn.execute(
                    "UPDATE chunks SET text=?, text_len=? WHERE chunk_id=?",
                    (title, len(title), chunk_id),
                )
                conn.execute(
                    "DELETE FROM chunk_embeddings WHERE chunk_id=? AND model=?",
                    (chunk_id, str(model)),
                )
        else:
            cur = conn.execute(
                "INSERT INTO chunks(page_id, start_line_no, end_line_no, text, text_len) "
                "VALUES(?, 0, 0, ?, ?)",
                (page_id, title, len(title)),
            )
            chunk_id = int(cur.lastrowid)

        # 3) Embedding для этой модели
        row = conn.execute(
            "SELECT 1 FROM chunk_embeddings WHERE chunk_id=? AND model=?",
            (chunk_id, str(model)),
        ).fetchone()
        if row:
            reused += 1
        else:
            pending_texts.append(title)
            pending_chunk_ids.append(chunk_id)

    conn.commit()

    print(f"Title pass: {reused} existing, {len(pending_texts)} to embed.")

    BATCH = max(1, int(batch_size))
    for i in range(0, len(pending_texts), BATCH):
        batch_t = pending_texts[i : i + BATCH]
        batch_c = pending_chunk_ids[i : i + BATCH]
        vecs = embedder.embed_texts(batch_t)
        if len(vecs) != len(batch_t):
            raise RuntimeError(
                f"Embedder returned {len(vecs)} vectors for {len(batch_t)} titles"
            )
        for cid, v in zip(batch_c, vecs):
            _upsert_embedding(conn, chunk_id=int(cid), model=str(model), vec=v)
            stats.embeddings_added += 1
        conn.commit()

    print(f"Title pass: done ({len(pending_texts)} new title embeddings).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step D (Index-time): build semantic chunks and embeddings into SQLite."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path (default: <project_root>/data/index.sqlite)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing chunks (and cascade delete chunk embeddings) before building.",
    )
    parser.add_argument(
        "--no-reuse-chunks",
        action="store_true",
        help="Do not reuse existing chunks; always insert new rows (NOT recommended).",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Always (re)embed chunks even if embeddings for this model exist.",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Only process first N pages (debug).",
    )

    # Chunking
    parser.add_argument(
        "--window-lines", type=int, default=5, help="Lines per chunk window."
    )
    parser.add_argument("--stride", type=int, default=2, help="Stride between windows.")
    parser.add_argument(
        "--min-line-len", type=int, default=2, help="Min line length to keep."
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=80,
        help="Min chunk size in characters to embed/store.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size."
    )

    # Embedder selection
    parser.add_argument(
        "--embedder",
        choices=["auto", "ollama", "sentence_transformers", "hash"],
        default="auto",
        help="Embedding backend. Default: auto (ollama -> sentence_transformers -> hash).",
    )

    # Ollama settings
    parser.add_argument(
        "--ollama-model", default="mxbai-embed-large", help="Ollama embedding model."
    )
    parser.add_argument(
        "--ollama-base-url", default="http://localhost:11434", help="Ollama base URL."
    )
    parser.add_argument(
        "--ollama-timeout-s", type=int, default=60, help="Ollama timeout seconds."
    )

    # sentence-transformers settings
    parser.add_argument(
        "--st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name/path.",
    )
    parser.add_argument(
        "--st-device", default=None, help="Sentence-transformers device (cpu/cuda)."
    )
    parser.add_argument("--st-batch-size", type=int, default=32, help="ST batch size.")

    # hash fallback settings (dev only)
    parser.add_argument(
        "--hash-dim", type=int, default=384, help="Hash embedder vector dimension."
    )
    parser.add_argument(
        "--hash-seed",
        default="older.index.hash_embedder.v1",
        help="Hash embedder seed (stability).",
    )

    args = parser.parse_args()

    db_path = Path(args.db) if args.db else (_PROJECT_ROOT / "data" / "index.sqlite")
    if not db_path.exists():
        print(
            f"[error] DB not found: {db_path}\n"
            "Run index-time first:\n"
            "  python scripts/step_a_scan_corpus.py\n"
            "  python scripts/step_b_index_pages.py",
            file=sys.stderr,
        )
        return 2

    # Pick embedder
    preferred = []
    if args.embedder == "auto":
        preferred = ["ollama", "sentence_transformers", "hash"]
    else:
        preferred = [args.embedder]

    try:
        embedder = create_embedder(
            preferred=preferred,  # type: ignore[arg-type]
            # ollama
            ollama_model=str(args.ollama_model),
            ollama_base_url=str(args.ollama_base_url),
            ollama_timeout_s=int(args.ollama_timeout_s),
            # sentence-transformers
            st_model=str(args.st_model),
            st_device=str(args.st_device) if args.st_device else None,
            st_batch_size=int(args.st_batch_size),
            # hash fallback
            hash_dim=int(args.hash_dim),
            hash_seed=str(args.hash_seed),
            # common
            normalized=True,
        )
    except EmbeddingsError as e:
        print(f"[error] Failed to create embedder: {e}", file=sys.stderr)
        return 2

    model = _model_label(embedder)

    stats = RunStats()
    t0 = time.time()

    conn = connect(db_path)
    try:
        init_db(conn)  # ensures schema.sql (including chunks tables) is applied
        _ensure_semantic_tables(conn)
        _validate_semantic_schema(conn)
        conn.commit()

        if args.clear:
            with transaction(conn):
                _clear_semantic(conn)
            conn.commit()

        reuse_chunks = not bool(args.no_reuse_chunks)
        incremental = not bool(args.no_incremental)

        print(f"Project root: {_PROJECT_ROOT}")
        print(f"DB path:      {db_path}")
        print(f"Embedder:     {embedder.info().backend}  model={model!r}")
        print(
            f"Chunking:     window_lines={args.window_lines}, stride={args.stride}, "
            f"min_line_len={args.min_line_len}, min_chunk_chars={args.min_chunk_chars}"
        )
        print(
            f"Behavior:     clear={bool(args.clear)}, reuse_chunks={reuse_chunks}, incremental={incremental}, "
            f"limit_pages={args.limit_pages}"
        )
        print("----")

        # Build everything
        build_chunks_and_embeddings(
            conn,
            embedder=embedder,
            model=str(model),
            window_lines=int(args.window_lines),
            stride=int(args.stride),
            min_line_len=int(args.min_line_len),
            min_chunk_chars=int(args.min_chunk_chars),
            batch_size=int(args.batch_size),
            incremental=bool(incremental),
            reuse_chunks=bool(reuse_chunks),
            limit_pages=args.limit_pages,
            stats=stats,
        )

        # Title pass: один embedding на имя каждого файла
        print("----")
        build_title_chunks_and_embeddings(
            conn,
            embedder=embedder,
            model=str(model),
            batch_size=int(args.batch_size),
            stats=stats,
        )

        stats.seconds = float(time.time() - t0)

        # Summary stats
        print("----")
        print("Done.")
        print(f"Pages seen:            {stats.pages_seen}")
        print(f"Pages skipped(no lines): {stats.pages_skipped_no_lines}")
        print(f"Chunks seen:           {stats.chunks_seen}")
        print(f"Chunks inserted:       {stats.chunks_inserted}")
        print(f"Chunks reused:         {stats.chunks_reused}")
        print(f"Embeddings added:      {stats.embeddings_added}")
        print(f"Embeddings skipped(existing): {stats.embeddings_skipped_existing}")
        print(f"Embedding dim:         {stats.embedding_dim}")
        print(f"Seconds:              {stats.seconds:.2f}")

        # Sanity: counts in DB for this model
        row = conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?",
            (str(model),),
        ).fetchone()
        count_model = int(row[0]) if row else 0
        print(f"DB embeddings for model: {count_model}")

        return 0
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
