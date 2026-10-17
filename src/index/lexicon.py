from __future__ import annotations

"""
Corpus lexicon builder and storage (Index-time artifact used at Query-time).

Purpose
-------
This module builds a "lexicon" (a list of known words from the indexed corpus)
to support Query-time typo correction (Указания.md шаг 2).

Key design goals:
- Domain-safe: we only collect *alphabetic* tokens (ru/en letters), so part numbers
  and technical identifiers (LM2596, 3.3V, I/O, VCC, etc.) are not part of the
  lexicon and therefore won't be "corrected" by typo logic.
- Incremental-friendly: can be (re)built from the existing `lines` table.
- Dependency-free: stdlib only. (Rapidfuzz is used by the typo-correction layer,
  not by this module.)

Data model
----------
We store a simple table:

  lexicon(term TEXT PRIMARY KEY, freq INTEGER NOT NULL)

Where:
- term is a normalized, case-folded word (Unicode letters only).
- freq is the total occurrence count across all indexed lines.

This table is safe to create lazily at Query-time:
- It depends only on Step B being run (so `lines` exists and is populated).
- It does not require changing existing tables.

Meta keys (optional)
--------------------
We record minimal metadata in `meta` if present:
- lexicon_version
- lexicon_min_len
- lexicon_built_at
"""

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import text_clean

LEXICON_VERSION = 1

# Token pattern: "letters only" across Unicode.
# Explanation: [^\W\d_] means "word characters excluding digits and underscore",
# which (in Unicode mode) corresponds mostly to letters (including Cyrillic).
_ALPHA_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class LexiconBuildStats:
    lines_scanned: int
    tokens_seen: int
    unique_terms_upserted: int
    seconds: float


def _now_s() -> int:
    return int(time.time())


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        # meta table might not exist if someone created a DB manually.
        return None
    if row is None:
        return None
    return str(row[0])


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    try:
        conn.execute(
            """
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
    except sqlite3.OperationalError:
        # meta table might not exist; ignore silently.
        return


def ensure_lexicon_table(conn: sqlite3.Connection) -> None:
    """
    Create the lexicon table if it doesn't exist.

    This is safe to call multiple times.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lexicon (
          term TEXT PRIMARY KEY,
          freq INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lexicon_freq ON lexicon(freq)")


def clear_lexicon(conn: sqlite3.Connection) -> None:
    """Delete all lexicon entries (keeps the table)."""
    ensure_lexicon_table(conn)
    conn.execute("DELETE FROM lexicon")


def lexicon_size(conn: sqlite3.Connection) -> int:
    """Return number of terms in lexicon table (0 if missing)."""
    try:
        row = conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def _normalize_term(term: str) -> str:
    """
    Normalize a term for lexicon storage / lookup.

    - Apply the same unicode normalization as the index-time pipeline
    - Case-fold (better than lower() for Unicode)
    """
    t = text_clean.normalize_unicode(term or "")
    t = t.casefold().strip()
    return t


def iter_alpha_terms(text: str, *, min_len: int = 4) -> Iterator[str]:
    """
    Extract normalized alphabetic terms (ru/en letters) from text.

    This intentionally ignores numbers and most punctuation so the lexicon stays
    "ordinary language" oriented (safe for typo correction).
    """
    if not text:
        return
    # Normalize first so µ/μ etc are stable; also removes weird spaces.
    norm = text_clean.normalize_unicode(text)
    for m in _ALPHA_TOKEN_RE.finditer(norm):
        raw = m.group(0)
        if not raw:
            continue
        t = _normalize_term(raw)
        if len(t) < int(min_len):
            continue
        yield t


def _require_tables_for_build(conn: sqlite3.Connection) -> None:
    # Step B is required (lines table must exist).
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lines'"
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Cannot build lexicon: missing table `lines`. "
            "Run index-time first (Step A + Step B)."
        )


def build_lexicon_from_lines(
    conn: sqlite3.Connection,
    *,
    min_len: int = 4,
    batch_lines: int = 5000,
    commit_every_batches: int = 10,
    where_sql: str = "",
    where_params: Sequence[object] = (),
) -> LexiconBuildStats:
    """
    Build/refresh lexicon terms by scanning the `lines` table.

    This function UPSERTs into `lexicon`, accumulating frequencies.

    Args:
        min_len: minimum word length to keep (default 4).
        batch_lines: number of `lines` rows to process per batch fetch.
        commit_every_batches: commit frequency to avoid huge transactions.
        where_sql: optional SQL fragment appended to the `SELECT ... FROM lines` query
                   (must start with "WHERE ..." or "JOIN ..." etc. Use carefully).
        where_params: parameters for where_sql.

    Returns:
        LexiconBuildStats with basic build metrics.
    """
    _require_tables_for_build(conn)
    ensure_lexicon_table(conn)

    t0 = time.time()

    lines_scanned = 0
    tokens_seen = 0
    unique_terms_upserted = 0

    # Use a stable ordering to make the process deterministic.
    base_sql = "SELECT text FROM lines "
    sql = (
        base_sql
        + (where_sql.strip() + " " if where_sql.strip() else "")
        + "ORDER BY line_id"
    )
    cur = conn.execute(sql, tuple(where_params))

    batch_no = 0
    while True:
        rows = cur.fetchmany(int(batch_lines))
        if not rows:
            break

        batch_no += 1
        counts: Dict[str, int] = {}

        for (text,) in rows:
            lines_scanned += 1
            for term in iter_alpha_terms(str(text or ""), min_len=int(min_len)):
                tokens_seen += 1
                counts[term] = counts.get(term, 0) + 1

        if counts:
            # Upsert batch counts.
            # We do not store enormous text fields; only term + batch freq.
            payload = [(term, freq) for term, freq in counts.items()]
            unique_terms_upserted += len(payload)

            conn.executemany(
                """
                INSERT INTO lexicon(term, freq) VALUES(?, ?)
                ON CONFLICT(term) DO UPDATE SET freq = freq + excluded.freq
                """,
                payload,
            )

        # Commit periodically to keep WAL size sane on big corpora.
        if commit_every_batches > 0 and (batch_no % int(commit_every_batches) == 0):
            conn.commit()

    conn.commit()

    # Meta (best-effort)
    _set_meta(conn, "lexicon_version", str(LEXICON_VERSION))
    _set_meta(conn, "lexicon_min_len", str(int(min_len)))
    _set_meta(conn, "lexicon_built_at", str(_now_s()))
    conn.commit()

    return LexiconBuildStats(
        lines_scanned=lines_scanned,
        tokens_seen=tokens_seen,
        unique_terms_upserted=unique_terms_upserted,
        seconds=float(time.time() - t0),
    )


def ensure_lexicon(
    conn: sqlite3.Connection,
    *,
    min_len: int = 4,
    force_rebuild: bool = False,
) -> Optional[LexiconBuildStats]:
    """
    Ensure lexicon exists and is populated.

    Behavior:
      - If force_rebuild=True: clears and rebuilds.
      - Else: if lexicon table is empty, builds it.
      - Else: does nothing.

    Returns:
      - LexiconBuildStats if a build happened
      - None if lexicon already existed and was non-empty
    """
    ensure_lexicon_table(conn)

    if force_rebuild:
        clear_lexicon(conn)
        conn.commit()
        return build_lexicon_from_lines(conn, min_len=int(min_len))

    if lexicon_size(conn) > 0:
        return None

    return build_lexicon_from_lines(conn, min_len=int(min_len))


def load_lexicon_terms(
    conn: sqlite3.Connection,
    *,
    min_freq: int = 2,
    limit: Optional[int] = None,
    order_by: str = "freq_desc",
) -> List[str]:
    """
    Load lexicon terms into memory.

    This is useful for typo correction libraries that need a "choices" list.

    Args:
        min_freq: drop extremely rare terms to reduce noise.
        limit: optional cap (for memory control).
        order_by: 'freq_desc' or 'term_asc'

    Returns:
        List of normalized terms.
    """
    ensure_lexicon_table(conn)

    min_freq = int(min_freq)
    if min_freq < 1:
        min_freq = 1

    if order_by not in ("freq_desc", "term_asc"):
        raise ValueError("order_by must be 'freq_desc' or 'term_asc'")

    order_sql = (
        "ORDER BY freq DESC, term ASC"
        if order_by == "freq_desc"
        else "ORDER BY term ASC"
    )

    lim_sql = ""
    params: List[object] = [min_freq]
    if limit is not None:
        lim_sql = " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(
        f"SELECT term FROM lexicon WHERE freq >= ? {order_sql}{lim_sql}",
        tuple(params),
    ).fetchall()

    return [str(r[0]) for r in rows]


def term_exists(conn: sqlite3.Connection, term: str) -> bool:
    """Check if a normalized term exists in lexicon."""
    ensure_lexicon_table(conn)
    t = _normalize_term(term)
    row = conn.execute("SELECT 1 FROM lexicon WHERE term = ? LIMIT 1", (t,)).fetchone()
    return row is not None
