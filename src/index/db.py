from __future__ import annotations

import contextlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional, Sequence

DocumentStatus = Literal["new", "changed", "unchanged", "indexed", "error"]


@dataclass(frozen=True)
class DocumentRow:
    doc_id: int
    rel_path: str
    mtime_ns: int
    size_bytes: int
    status: str
    last_error: Optional[str]
    indexed_at: Optional[int]


# The database schema lives in `schema.sql` next to this module.
# Step B can extend that file and (optionally) bump SCHEMA_VERSION with migrations if needed.
SCHEMA_VERSION = 1


def _load_schema_sql() -> str:
    schema_path = Path(__file__).with_name("schema.sql")
    return schema_path.read_text(encoding="utf-8")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a SQLite connection configured for this project.

    Notes:
      - We rely on WAL for better write performance.
      - Row factory is enabled to allow name-based access.
    """
    path = str(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create/update the DB schema to the latest version supported by this code.

    Step A creates only `meta` and `documents`.
    Step B will add more tables and bump `SCHEMA_VERSION`.
    """
    conn.executescript(_load_schema_sql())

    current_version = get_meta(conn, "schema_version")
    if current_version is None:
        set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    else:
        # Future-proof: if you bump schema later, implement migrations here.
        # For now, we only support version 1.
        if int(current_version) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported schema_version in DB: {current_version}. "
                f"Expected: {SCHEMA_VERSION}. You may need to run migrations."
            )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return str(row["value"])


def get_document_by_rel_path(
    conn: sqlite3.Connection, rel_path: str
) -> Optional[DocumentRow]:
    row = conn.execute(
        """
        SELECT doc_id, rel_path, mtime_ns, size_bytes, status, last_error, indexed_at
        FROM documents
        WHERE rel_path = ?
        """,
        (rel_path,),
    ).fetchone()
    if row is None:
        return None
    return DocumentRow(
        doc_id=int(row["doc_id"]),
        rel_path=str(row["rel_path"]),
        mtime_ns=int(row["mtime_ns"]),
        size_bytes=int(row["size_bytes"]),
        status=str(row["status"]),
        last_error=row["last_error"],
        indexed_at=row["indexed_at"],
    )


def list_documents_by_status(
    conn: sqlite3.Connection, statuses: Sequence[DocumentStatus]
) -> list[DocumentRow]:
    if not statuses:
        return []

    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"""
        SELECT doc_id, rel_path, mtime_ns, size_bytes, status, last_error, indexed_at
        FROM documents
        WHERE status IN ({placeholders})
        ORDER BY rel_path
        """,
        tuple(statuses),
    ).fetchall()

    return [
        DocumentRow(
            doc_id=int(r["doc_id"]),
            rel_path=str(r["rel_path"]),
            mtime_ns=int(r["mtime_ns"]),
            size_bytes=int(r["size_bytes"]),
            status=str(r["status"]),
            last_error=r["last_error"],
            indexed_at=r["indexed_at"],
        )
        for r in rows
    ]


def _now_s() -> int:
    return int(time.time())


def update_document_status(
    conn: sqlite3.Connection, doc_id: int, status: DocumentStatus
) -> None:
    conn.execute(
        """
        UPDATE documents
           SET status = ?,
               updated_at = CAST(strftime('%s','now') AS INTEGER)
         WHERE doc_id = ?
        """,
        (status, doc_id),
    )


def mark_document_indexed(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute(
        """
        UPDATE documents
           SET status     = 'indexed',
               indexed_at = ?,
               last_error = NULL,
               updated_at = CAST(strftime('%s','now') AS INTEGER)
         WHERE doc_id = ?
        """,
        (_now_s(), doc_id),
    )


def mark_document_error(conn: sqlite3.Connection, doc_id: int, error: str) -> None:
    conn.execute(
        """
        UPDATE documents
           SET status     = 'error',
               last_error = ?,
               updated_at = CAST(strftime('%s','now') AS INTEGER)
         WHERE doc_id = ?
        """,
        (error, doc_id),
    )


@dataclass(frozen=True)
class UpsertResult:
    doc_id: int
    action: Literal["inserted", "changed", "unchanged"]


def upsert_document_from_fs_stat(
    conn: sqlite3.Connection,
    rel_path_posix: str,
    mtime_ns: int,
    size_bytes: int,
) -> UpsertResult:
    """
    Insert a new document row, or update an existing one if the file changed.

    Returns:
      - action=inserted: new file discovered
      - action=changed: existing file stats differ
      - action=unchanged: stats match
    """
    existing = get_document_by_rel_path(conn, rel_path_posix)
    if existing is None:
        cur = conn.execute(
            """
            INSERT INTO documents(rel_path, mtime_ns, size_bytes, status)
            VALUES(?, ?, ?, 'new')
            """,
            (rel_path_posix, int(mtime_ns), int(size_bytes)),
        )
        return UpsertResult(doc_id=int(cur.lastrowid), action="inserted")

    if existing.mtime_ns != int(mtime_ns) or existing.size_bytes != int(size_bytes):
        conn.execute(
            """
            UPDATE documents
            SET mtime_ns=?, size_bytes=?, status='changed', last_error=NULL
            WHERE doc_id=?
            """,
            (int(mtime_ns), int(size_bytes), int(existing.doc_id)),
        )
        return UpsertResult(doc_id=int(existing.doc_id), action="changed")

    # File is unchanged on disk.
    #
    # IMPORTANT for Step B queueing:
    # - If a document was never indexed (indexed_at IS NULL), it must remain queued for indexing.
    # - If it is already indexed, keep status='indexed' (so Step B can skip it).
    # - If it is currently 'new' or 'changed' (or even 'error'), keep that status (still queued).
    #
    # This prevents Step A from accidentally turning the whole corpus into 'unchanged'
    # before Step B has indexed it.
    if existing.status == "indexed" or existing.indexed_at is not None:
        # Keep indexed documents indexed.
        return UpsertResult(doc_id=int(existing.doc_id), action="unchanged")

    if existing.status in ("new", "changed", "error"):
        # Keep queued/error statuses as-is.
        return UpsertResult(doc_id=int(existing.doc_id), action="unchanged")

    # Status is typically 'unchanged' here.
    # If the document was never indexed, re-queue it as 'new' so Step B will pick it up.
    conn.execute(
        "UPDATE documents SET status='new' WHERE doc_id=?",
        (int(existing.doc_id),),
    )

    return UpsertResult(doc_id=int(existing.doc_id), action="unchanged")


def get_corpus_root(conn: sqlite3.Connection) -> Optional[Path]:
    """
    Returns the corpus root directory stored in meta['corpus_root'] (Step A writes it).

    Step B uses this to resolve absolute file paths:
      abs_path = corpus_root / rel_path
    """
    value = get_meta(conn, "corpus_root")
    if value is None:
        return None
    return Path(value)


def normalize_rel_path(rel_path: str) -> str:
    """
    Normalize relative paths stored in DB to POSIX style.
    This avoids Windows backslash issues when building sources/citations.
    """
    return rel_path.replace("\\", "/")


def delete_pages_for_doc(conn: sqlite3.Connection, doc_id: int) -> None:
    """
    Delete all page records for a document.

    Because `pages.doc_id` has ON DELETE CASCADE to `lines`, deleting pages
    automatically deletes all associated lines as well.

    This is required for Step B re-indexing of documents with status='changed'.
    """
    conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))


def insert_page(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    page_no: int,
    text: str,
    is_ocr: int = 0,
    text_hash: Optional[str] = None,
) -> int:
    """
    Insert one page of extracted text.

    `page_no` is 1-based for user-facing citations.
    Returns the new `page_id`.
    """
    t = text or ""
    cur = conn.execute(
        """
        INSERT INTO pages(doc_id, page_no, text, is_ocr, text_len, text_hash)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (doc_id, int(page_no), t, int(is_ocr), len(t), text_hash),
    )
    return int(cur.lastrowid)


def insert_lines_bulk(
    conn: sqlite3.Connection,
    *,
    page_id: int,
    lines: Sequence[str],
    min_len: int = 1,
) -> int:
    """
    Insert many lines for a page efficiently.

    Lines are stored 1-based (`line_no` starts at 1).
    Returns the number of inserted lines.
    """
    rows: list[tuple[int, int, str, int]] = []
    line_no = 0
    for raw in lines:
        s = (raw or "").strip()
        if len(s) < min_len:
            continue
        line_no += 1
        rows.append((int(page_id), int(line_no), s, len(s)))

    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO lines(page_id, line_no, text, text_len)
        VALUES(?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """
    Simple transaction context manager.

    Usage:
      with transaction(conn):
          ...
    """
    try:
        conn.execute("BEGIN")
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
