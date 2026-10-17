from __future__ import annotations

"""
Step B: index PDF content into SQLite tables `pages` and `lines`.

This script is the "content indexing" part of Index-time.

Inputs
------
- A corpus folder with PDFs (default: <project_root>/corpus)
- A SQLite DB created/maintained by Step A (scripts/step_a_scan_corpus.py)

What it does
------------
For each document in chosen statuses (default: new, changed):
  1) Resolve PDF path: <corpus_root>/<rel_path>
  2) Extract per-page text using PyMuPDF (fitz)
  3) (Optional) OCR pages where extracted text is too short
  4) Normalize text using src.index.text_clean
  5) Store into DB:
     - pages(doc_id, page_no, text, is_ocr, text_len, text_hash)
     - lines(page_id, line_no, text, text_len)
  6) Mark document as indexed

How it interacts with Step A
----------------------------
- Step A scans filesystem and maintains documents table with statuses:
    new, changed, indexed, unchanged, error
- Step B should normally process: new, changed
  (You can pass --statuses unchanged if needed for debugging.)

Dependencies
------------
Required:
- pymupdf  (imported as `fitz`)

Optional (only if --ocr):
- pillow
- pytesseract
- Tesseract OCR installed in the OS + language packs (e.g. rus, eng)

Run
---
From project root:
  python scripts/step_b_index_pages.py

Typical:
  python scripts/step_b_index_pages.py --ocr --text-threshold 60 --dpi 250 --ocr-lang rus+eng --ocr-psm 6
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Allow running from any working directory: add project root (parent of scripts/) to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index import text_clean  # noqa: E402
from src.index.db import (  # noqa: E402
    connect,
    delete_pages_for_doc,
    get_meta,
    init_db,
    insert_lines_bulk,
    insert_page,
    list_documents_by_status,
    mark_document_error,
    mark_document_indexed,
    transaction,
)
from src.index.extractors import get_extractor  # noqa: E402

# -----------------------------
# Stats / helpers
# -----------------------------


@dataclass
class RunStats:
    docs_total: int = 0
    docs_indexed: int = 0
    docs_error: int = 0
    pages_inserted: int = 0
    ocr_pages: int = 0
    lines_inserted: int = 0


def _now_s() -> int:
    return int(time.time())


def _resolve_corpus_root(conn, corpus_root_cli: Optional[str]) -> Path:
    if corpus_root_cli:
        return Path(corpus_root_cli).resolve()

    meta_root = get_meta(conn, "corpus_root")
    if meta_root:
        return Path(meta_root).resolve()

    # Fallback for cases where DB wasn't created by Step A.
    return (_PROJECT_ROOT / "corpus").resolve()


def _iter_lines_for_storage(text: str, *, min_len: int = 2) -> list[str]:
    """
    Split text into lines for the `lines` table.

    Prefer the shared text_clean.split_lines if present; otherwise fall back to a
    simple implementation.

    We intentionally keep line structure because datasheet "tables" often become
    line-based text after extraction/OCR.
    """
    split_fn = getattr(text_clean, "split_lines", None)
    if callable(split_fn):
        try:
            return list(split_fn(text, min_len=min_len))  # type: ignore[misc]
        except TypeError:
            # If signature differs, fall back below.
            pass

    out: list[str] = []
    for raw in (text or "").split("\n"):
        s = raw.strip()
        if len(s) >= min_len:
            out.append(s)
    return out


def _should_ocr(extracted_text: str, *, threshold: int) -> bool:
    return len((extracted_text or "").strip()) < int(threshold)


# -----------------------------
# OCR (optional)
# -----------------------------


def _import_pymupdf():
    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for Step B. Install with: pip install pymupdf. "
            f"Original import error: {e}"
        ) from e
    return fitz


def _import_ocr_deps():
    try:
        import pytesseract  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pytesseract is required for OCR. Install with: pip install pytesseract. "
            f"Original import error: {e}"
        ) from e

    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for OCR preprocessing. Install with: pip install pillow. "
            f"Original import error: {e}"
        ) from e

    return pytesseract, Image, ImageOps


def _render_page_to_pil(fitz, page, *, dpi: int):
    pytesseract, Image, ImageOps = _import_ocr_deps()  # noqa: F841
    pix = page.get_pixmap(dpi=int(dpi))
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


def _preprocess_for_ocr(img):
    """
    Lightweight preprocessing for technical scans (text + simple tables).

    This is intentionally cheap; you can improve later with deskew, denoise, etc.
    """
    _pytesseract, _Image, ImageOps = _import_ocr_deps()

    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)

    # Mild binarization often improves number readability. Keep simple.
    bw = g.point(lambda x: 255 if x > 180 else 0, mode="1")
    return bw.convert("L")


def _ocr_page_to_text(fitz, page, *, dpi: int, lang: str, psm: int) -> str:
    pytesseract, _Image, _ImageOps = _import_ocr_deps()

    img = _render_page_to_pil(fitz, page, dpi=int(dpi))
    img = _preprocess_for_ocr(img)

    config = f"--oem 3 --psm {int(psm)}"
    return pytesseract.image_to_string(img, lang=str(lang), config=config)


# -----------------------------
# Core processing
# -----------------------------


def _persist_page(
    *,
    conn,
    doc_id: int,
    page_no: int,
    text: str,
    is_ocr: int,
    store_text_hash: bool,
) -> int:
    """Insert one page and its lines. Returns count of inserted lines."""
    page_id = insert_page(
        conn,
        doc_id=int(doc_id),
        page_no=int(page_no),
        text=text,
        is_ocr=int(is_ocr),
        text_hash=None,
    )
    if store_text_hash:
        import hashlib
        h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        conn.execute(
            "UPDATE pages SET text_hash=? WHERE page_id=?", (h, int(page_id))
        )
    lines = _iter_lines_for_storage(text, min_len=2)
    return insert_lines_bulk(conn, page_id=int(page_id), lines=lines, min_len=2)


def _process_pdf(
    *,
    conn,
    fitz,
    abs_path: Path,
    doc_id: int,
    enable_ocr: bool,
    text_threshold: int,
    ocr_dpi: int,
    ocr_lang: str,
    ocr_psm: int,
    max_pages: Optional[int],
    store_text_hash: bool,
) -> tuple[int, int, int]:
    """Existing PDF logic with OCR fallback. Returns (pages, ocr_pages, lines)."""
    pages_inserted = 0
    ocr_pages = 0
    lines_inserted = 0

    pdf = fitz.open(str(abs_path))
    try:
        page_count = int(pdf.page_count)
        pages_to_process = page_count
        if max_pages is not None:
            pages_to_process = min(pages_to_process, int(max_pages))

        for i in range(pages_to_process):
            page = pdf.load_page(i)
            extracted = text_clean.clean_page_text(page.get_text("text") or "")
            final_text = extracted
            used_ocr = 0

            if enable_ocr and _should_ocr(extracted, threshold=text_threshold):
                try:
                    ocr_raw = _ocr_page_to_text(
                        fitz, page,
                        dpi=int(ocr_dpi), lang=str(ocr_lang), psm=int(ocr_psm),
                    )
                    final_text = text_clean.clean_ocr_text(ocr_raw)
                    used_ocr = 1
                    ocr_pages += 1
                except Exception:
                    final_text = final_text or ""
                    used_ocr = 0

            lines_inserted += _persist_page(
                conn=conn, doc_id=doc_id, page_no=i + 1,
                text=final_text, is_ocr=used_ocr,
                store_text_hash=store_text_hash,
            )
            pages_inserted += 1
    finally:
        pdf.close()

    return pages_inserted, ocr_pages, lines_inserted


def _process_via_extractor(
    *,
    conn,
    extractor,
    abs_path: Path,
    doc_id: int,
    store_text_hash: bool,
) -> tuple[int, int, int]:
    """Non-PDF formats (TXT, MD, DOCX). Returns (pages, ocr_pages, lines)."""
    pages_inserted = 0
    ocr_pages = 0
    lines_inserted = 0

    extracted_pages = extractor.extract(abs_path)
    for ep in extracted_pages:
        if not (ep.text or "").strip():
            continue
        lines_inserted += _persist_page(
            conn=conn, doc_id=doc_id, page_no=int(ep.page_no),
            text=ep.text, is_ocr=int(bool(ep.is_ocr)),
            store_text_hash=store_text_hash,
        )
        pages_inserted += 1
        if ep.is_ocr:
            ocr_pages += 1

    return pages_inserted, ocr_pages, lines_inserted


def _process_document(
    *,
    conn,
    fitz,
    corpus_root: Path,
    doc_id: int,
    rel_path: str,
    status: str,
    enable_ocr: bool,
    text_threshold: int,
    ocr_dpi: int,
    ocr_lang: str,
    ocr_psm: int,
    max_pages: Optional[int],
    store_text_hash: bool,
) -> tuple[int, int, int]:
    """
    Dispatches by file extension: PDF goes through PyMuPDF with optional OCR;
    other supported formats (TXT, MD, DOCX) go through the extractor system.

    Returns: (pages_inserted, ocr_pages, lines_inserted)
    """
    abs_path = (corpus_root / rel_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"File not found: {abs_path}")

    # Re-index strategy for changed docs: delete old pages (lines cascade).
    if status == "changed":
        delete_pages_for_doc(conn, int(doc_id))

    ext = abs_path.suffix.lower()
    if ext == ".pdf":
        return _process_pdf(
            conn=conn, fitz=fitz, abs_path=abs_path, doc_id=doc_id,
            enable_ocr=enable_ocr, text_threshold=text_threshold,
            ocr_dpi=ocr_dpi, ocr_lang=ocr_lang, ocr_psm=ocr_psm,
            max_pages=max_pages, store_text_hash=store_text_hash,
        )

    extractor = get_extractor(abs_path)
    if extractor is None:
        raise RuntimeError(f"No extractor registered for extension {ext!r}")

    return _process_via_extractor(
        conn=conn, extractor=extractor, abs_path=abs_path,
        doc_id=doc_id, store_text_hash=store_text_hash,
    )


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step B: extract text from PDFs (PyMuPDF + optional OCR) into pages/lines tables."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path (default: <project_root>/data/index.sqlite)",
    )
    parser.add_argument(
        "--corpus-root",
        default=None,
        help="Corpus folder with PDFs (default from meta or <project_root>/corpus)",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["new", "changed"],
        help="Document statuses to process (default: new changed)",
    )

    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for pages with too little extracted text",
    )
    parser.add_argument(
        "--text-threshold",
        type=int,
        default=60,
        help="If extracted text length < threshold, try OCR (default: 60)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI for OCR rendering (default: 250)",
    )
    parser.add_argument(
        "--ocr-lang",
        default="rus+eng",
        help="Tesseract languages (default: rus+eng)",
    )
    parser.add_argument(
        "--ocr-psm",
        type=int,
        default=6,
        help="Tesseract PSM mode (default: 6)",
    )

    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only first N documents (debug)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Process only first N pages per PDF (debug)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first document error",
    )
    parser.add_argument(
        "--store-text-hash",
        action="store_true",
        help="Store SHA1(text) into pages.text_hash (debug/dedup support)",
    )

    args = parser.parse_args()

    # Required dependency (text extraction)
    fitz = _import_pymupdf()

    # OCR deps are optional unless --ocr is set.
    if args.ocr:
        _import_ocr_deps()

    db_path = Path(args.db) if args.db else (_PROJECT_ROOT / "data" / "index.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    stats = RunStats()

    conn = connect(db_path)
    try:
        init_db(conn)
        conn.commit()

        corpus_root = _resolve_corpus_root(conn, args.corpus_root)
        if not corpus_root.exists():
            print(f"[error] corpus_root not found: {corpus_root}", file=sys.stderr)
            return 2

        docs = list_documents_by_status(conn, statuses=args.statuses)
        if args.limit_docs is not None:
            docs = docs[: int(args.limit_docs)]

        stats.docs_total = len(docs)

        print(f"Project root: {_PROJECT_ROOT}")
        print(f"Corpus root:  {corpus_root}")
        print(f"DB path:      {db_path}")
        print(f"To process:   {stats.docs_total} documents (statuses={args.statuses})")
        print(
            f"OCR enabled:  {bool(args.ocr)} "
            f"(threshold={args.text_threshold}, dpi={args.dpi}, lang={args.ocr_lang}, psm={args.ocr_psm})"
        )
        print("----")

        for idx, doc in enumerate(docs, start=1):
            print(
                f"[{idx}/{stats.docs_total}] doc_id={doc.doc_id} status={doc.status} path={doc.rel_path}"
            )

            try:
                with transaction(conn):
                    pages_inserted, ocr_pages, lines_inserted = _process_document(
                        conn=conn,
                        fitz=fitz,
                        corpus_root=corpus_root,
                        doc_id=int(doc.doc_id),
                        rel_path=str(doc.rel_path),
                        status=str(doc.status),
                        enable_ocr=bool(args.ocr),
                        text_threshold=int(args.text_threshold),
                        ocr_dpi=int(args.dpi),
                        ocr_lang=str(args.ocr_lang),
                        ocr_psm=int(args.ocr_psm),
                        max_pages=args.max_pages,
                        store_text_hash=bool(args.store_text_hash),
                    )
                    mark_document_indexed(conn, int(doc.doc_id))

                stats.docs_indexed += 1
                stats.pages_inserted += pages_inserted
                stats.ocr_pages += ocr_pages
                stats.lines_inserted += lines_inserted

                print(f"  indexed: pages={pages_inserted}, ocr_pages={ocr_pages}, lines={lines_inserted}")

            except Exception as e:
                stats.docs_error += 1
                err = f"{type(e).__name__}: {e}"
                print(f"  [error] {err}", file=sys.stderr)

                # Record error outside the failed transaction
                try:
                    mark_document_error(conn, int(doc.doc_id), err)
                    conn.commit()
                except Exception as e2:  # pragma: no cover
                    print(f"  [error] failed to record error in DB: {e2}", file=sys.stderr)

                if args.fail_fast:
                    break

        print("----")
        print("Done.")
        print(f"Docs processed: {stats.docs_total}")
        print(f"Docs indexed:   {stats.docs_indexed}")
        print(f"Docs errors:    {stats.docs_error}")
        print(f"Pages inserted: {stats.pages_inserted}")
        print(f"OCR pages:      {stats.ocr_pages}")
        print(f"Lines inserted: {stats.lines_inserted}")

        return 0 if stats.docs_error == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())