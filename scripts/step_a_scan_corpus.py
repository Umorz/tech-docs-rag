from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set

# Allow running this script from any working directory:
# we add the project root (parent of `scripts/`) to sys.path so `import src.*` works.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index.db import (
    connect,
    get_meta,
    init_db,
    set_meta,
    transaction,
    upsert_document_from_fs_stat,
)
from src.index.extractors import supported_extensions


@dataclass(frozen=True)
class ScanStats:
    pdf_found: int
    inserted_new: int
    marked_changed: int
    unchanged: int
    missing_marked: int


def _iter_supported_files(corpus_root: Path) -> Iterable[Path]:
    """Yield all files in corpus_root with a supported extension."""
    exts = set(supported_extensions())  # e.g. {".pdf", ".txt", ".md", ".docx"}
    for p in corpus_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _default_project_root() -> Path:
    # scripts/step_a_scan_corpus.py -> project root = parents[1]
    return Path(__file__).resolve().parents[1]


def _mark_missing_as_error(db_path: Path, rel_paths: Set[str]) -> int:
    """
    Step A doesn't delete anything. If a file disappeared from the corpus folder,
    we mark it as status=error with last_error describing the situation.

    This keeps statuses consistent with the shared DB module:
      new|changed|unchanged|indexed|error
    """
    if not rel_paths:
        return 0

    conn = connect(db_path)
    try:
        placeholders = ",".join("?" for _ in rel_paths)
        now_s = int(time.time())
        conn.execute(
            f"""
            UPDATE documents
               SET status='error',
                   last_error='missing from filesystem during corpus scan'
             WHERE rel_path IN ({placeholders})
            """,
            tuple(sorted(rel_paths)),
        )
        conn.commit()
        return len(rel_paths)
    finally:
        conn.close()


def scan_corpus(
    corpus_root: Path, db_path: Path, prune_missing: bool, dry_run: bool
) -> ScanStats:
    corpus_root = corpus_root.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(_iter_supported_files(corpus_root))
    seen_now: Set[str] = set()

    conn = connect(db_path)
    try:
        init_db(conn)

        prev_root = get_meta(conn, "corpus_root")
        if prev_root is None:
            set_meta(conn, "corpus_root", str(corpus_root))
        else:
            if Path(prev_root).resolve() != corpus_root:
                print(
                    f"[warn] corpus_root changed in meta:\n"
                    f"  prev: {prev_root}\n"
                    f"  now:  {corpus_root}\n"
                    f"DB stores rel_path, so this is usually OK as long as the relative layout didn't change.",
                    file=sys.stderr,
                )
                set_meta(conn, "corpus_root", str(corpus_root))

        # Commit schema/meta changes separately, so later dry-run rollbacks don't revert them.
        conn.commit()

        # Load current rel_path set for optional prune_missing.
        existing_rel_paths: Set[str] = set()
        if prune_missing:
            rows = conn.execute("SELECT rel_path FROM documents").fetchall()
            existing_rel_paths = {str(r["rel_path"]) for r in rows}

        inserted_new = 0
        marked_changed = 0
        unchanged = 0

        def _process_files() -> None:
            nonlocal inserted_new, marked_changed, unchanged
            for abs_path in pdf_paths:
                rel_path = abs_path.relative_to(corpus_root).as_posix()
                seen_now.add(rel_path)

                st = abs_path.stat()
                res = upsert_document_from_fs_stat(
                    conn=conn,
                    rel_path_posix=rel_path,
                    mtime_ns=int(st.st_mtime_ns),
                    size_bytes=int(st.st_size),
                )

                if res.action == "inserted":
                    inserted_new += 1
                elif res.action == "changed":
                    marked_changed += 1
                else:
                    unchanged += 1

        if dry_run:
            conn.execute("BEGIN")
            try:
                _process_files()
            finally:
                conn.rollback()
        else:
            with transaction(conn):
                _process_files()

        missing_marked = 0
        if prune_missing and existing_rel_paths:
            missing_rel_paths = existing_rel_paths.difference(seen_now)
            if missing_rel_paths:
                if dry_run:
                    missing_marked = len(missing_rel_paths)
                else:
                    # We perform this update in a separate short-lived connection to keep logic simple.
                    missing_marked = _mark_missing_as_error(db_path, missing_rel_paths)

        return ScanStats(
            pdf_found=len(pdf_paths),
            inserted_new=inserted_new,
            marked_changed=marked_changed,
            unchanged=unchanged,
            missing_marked=missing_marked,
        )
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step A: scan corpus folder and register PDF documents in SQLite (no text extraction yet)."
    )
    parser.add_argument(
        "--corpus-root",
        default=None,
        help="Path to corpus folder with PDFs (default: <project_root>/corpus)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to sqlite db file (default: <project_root>/data/index.sqlite)",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Mark documents missing from filesystem as status=error (safe; does not delete rows).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write anything to DB; just print stats.",
    )

    args = parser.parse_args()

    project_root = _default_project_root()
    corpus_root = (
        Path(args.corpus_root) if args.corpus_root else (project_root / "corpus")
    )
    db_path = Path(args.db) if args.db else (project_root / "data" / "index.sqlite")

    if not corpus_root.exists():
        print(f"[error] corpus_root not found: {corpus_root}", file=sys.stderr)
        return 2

    stats = scan_corpus(
        corpus_root=corpus_root,
        db_path=db_path,
        prune_missing=bool(args.prune_missing),
        dry_run=bool(args.dry_run),
    )

    print(f"Project root: {project_root}")
    print(f"Corpus root:  {corpus_root.resolve()}")
    print(f"DB path:      {db_path.resolve()}")
    print(f"Files found:  {stats.pdf_found}")
    print(f"New:          {stats.inserted_new}")
    print(f"Changed:      {stats.marked_changed}")
    print(f"Unchanged:    {stats.unchanged}")
    if args.prune_missing:
        print(f"Missing marked: {stats.missing_marked}")
    if args.dry_run:
        print("(dry-run: no DB changes written)")

    # Helpful debug summary: show current DB counts by status after the scan.
    # This makes it easier to verify the Step A -> Step B queue behavior.
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM documents GROUP BY status ORDER BY status"
        ).fetchall()
        print("DB status counts:")
        for r in rows:
            print(f"  {r['status']}: {r['cnt']}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
