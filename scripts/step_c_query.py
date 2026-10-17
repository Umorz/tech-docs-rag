from __future__ import annotations

"""
Step C (Query-time, MVP): search indexed PDFs using SQLite FTS5 + BM25.

This script sits on top of the existing Index-time pipeline:

- Step A: scripts/step_a_scan_corpus.py  -> fills `documents`
- Step B: scripts/step_b_index_pages.py  -> fills `pages`/`lines` + FTS tables

This MVP implements (per Указания.md):
  (1) Формирование запроса  (minimal + safe, with an "advanced" pass-through mode)
  (4) Булев поиск / BM25    (SQLite FTS5 MATCH + bm25())

Future steps (typo correction, synonyms/morphology, embeddings, reranking) are
intentionally NOT implemented here yet, but the structure is designed to allow
adding them without breaking the CLI contract.

Usage examples (from project root):
  python scripts/step_c_query.py "absolute maximum ratings"
  python scripts/step_c_query.py --in pages "thermal resistance"
  python scripts/step_c_query.py --mode advanced '"buck converter" OR boost'
  python scripts/step_c_query.py --context 2 "VIN"

Notes:
  - For best citations, default target is `lines`.
  - If you have scanned PDFs, consider indexing with OCR first:
      python scripts/step_b_index_pages.py --ocr
"""

import argparse
import heapq
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

# Allow running from any working directory: add project root (parent of scripts/) to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index.db import connect  # noqa: E402
from src.index.embeddings import (  # noqa: E402
    EmbeddingsError,
    create_embedder,
    stable_model_label_for_embedder,
)
from src.index.hybrid import HybridOptions, hybrid_search_rrf  # noqa: E402
from src.index.search import (  # noqa: E402
    PageHit,
    SearchError,
    SearchOptions,
    fetch_line_context,
    search,
)


def _read_query_from_args_or_stdin(query_parts: list[str]) -> str:
    if query_parts:
        return " ".join(query_parts).strip()
    # If no args provided, read from stdin (useful for piping).
    data = sys.stdin.read()
    return (data or "").strip()


def _semantic_model_label(embedder) -> str:
    """
    Create a stable model label that matches what Step D stores in chunk_embeddings.model.

    Delegates to the shared helper so the label is consistent across:
      - Step D: scripts/step_d_embed_chunks.py (index-time embeddings)
      - Step C: scripts/step_c_query.py --semantic / --hybrid (query-time retrieval)

    This prevents subtle mismatches where embeddings exist in DB but query-time selects
    a different `chunk_embeddings.model` string.
    """
    return stable_model_label_for_embedder(embedder)


def _decode_f32(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a)


def _dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for x, y in zip(a, b, strict=False):
        s += float(x) * float(y)
    return s


def _l2_norm(vec: list[float]) -> float:
    s2 = 0.0
    for x in vec:
        s2 += float(x) * float(x)
    return s2**0.5 if s2 > 0.0 else 0.0


def _semantic_search_sql(
    conn,
    *,
    embedder,
    model_label: str,
    query: str,
    top_k: int,
    min_score: float,
    rel_path: Optional[str],
) -> list[tuple[float, int, str, int, int, int, str]]:
    """
    Semantic retrieval over chunk embeddings stored in SQLite.

    Returns list of:
      (score, chunk_id, rel_path, page_no, start_line_no, end_line_no, text)
    """
    q_vec = embedder.embed_query(query)
    if not isinstance(q_vec, list):
        q_vec = list(q_vec)

    info = embedder.info()
    assume_normalized = bool(getattr(info, "normalized", False))
    qn = 1.0 if assume_normalized else _l2_norm(q_vec)

    where = "WHERE e.model = ?"
    params: list[object] = [str(model_label)]
    if rel_path:
        where += " AND d.rel_path = ?"
        params.append(str(rel_path))

    rows = conn.execute(
        f"""
        SELECT
          c.chunk_id,
          d.rel_path,
          p.page_no,
          COALESCE(c.start_line_no, 0) AS start_line_no,
          COALESCE(c.end_line_no, 0)   AS end_line_no,
          c.text,
          e.dim,
          e.vector
        FROM chunk_embeddings e
        JOIN chunks c     ON c.chunk_id = e.chunk_id
        JOIN pages p      ON p.page_id  = c.page_id
        JOIN documents d  ON d.doc_id   = p.doc_id
        {where}
        """,
        tuple(params),
    ).fetchall()

    heap: list[tuple[float, tuple[float, int, str, int, int, int, str]]] = []

    for r in rows:
        chunk_id = int(r[0])
        rp = str(r[1])
        page_no = int(r[2])
        start_ln = int(r[3])
        end_ln = int(r[4])
        text = str(r[5])
        dim = int(r[6])
        blob = r[7]

        if dim != len(q_vec):
            continue
        if not isinstance(blob, (bytes, bytearray)):
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

        if float(score) < float(min_score):
            continue

        item = (float(score), chunk_id, rp, page_no, start_ln, end_ln, text)

        if top_k <= 0:
            continue

        if len(heap) < int(top_k):
            heapq.heappush(heap, (float(score), item))
        else:
            # Keep only top_k largest scores in a min-heap
            if float(score) > heap[0][0]:
                heapq.heapreplace(heap, (float(score), item))

    out = [it for _s, it in heap]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# -----------------------------
# Hybrid reranking (Step 6): lexical + semantic
# -----------------------------


@dataclass(frozen=True)
class _HybridKey:
    rel_path: str
    page_no: int
    start_line_no: int
    end_line_no: int


@dataclass(frozen=True)
class HybridHit:
    key: _HybridKey
    text: str

    # Signals (optional)
    lex_rank: Optional[int] = None
    lex_score: Optional[float] = None  # bm25 (lower is better) for debugging

    sem_rank: Optional[int] = None
    sem_score: Optional[float] = (
        None  # cosine similarity (higher is better) for debugging
    )

    # Final rerank score (higher is better)
    score: float = 0.0


def _rrf_score(rank: Optional[int], *, k: int, weight: float) -> float:
    """
    Reciprocal Rank Fusion component:
      weight / (k + rank)
    rank is 1-based. If rank is None -> contributes 0.
    """
    if rank is None:
        return 0.0
    r = int(rank)
    if r <= 0:
        return 0.0
    return float(weight) / float(int(k) + r)


def _hybrid_rerank_rrf(
    *,
    lexical_hits,
    semantic_hits: list[tuple[float, int, str, int, int, int, str]],
    rrf_k: int,
    w_lex: float,
    w_sem: float,
    limit: int,
    prefer_text: str = "lexical",
) -> list[HybridHit]:
    """
    Merge two hit lists and rerank via Reciprocal Rank Fusion (RRF).

    - lexical_hits: output of src.index.search.search() (LineHit or PageHit list)
    - semantic_hits: list of tuples from _semantic_search_sql:
        (score, chunk_id, rel_path, page_no, start_line_no, end_line_no, text)

    We use stable citation keys:
      - lexical line hit -> (rel_path, page_no, line_no, line_no)
      - lexical page hit -> (rel_path, page_no, 0, 0)
      - semantic chunk   -> (rel_path, page_no, start_line_no, end_line_no)

    Scoring:
      score = w_lex/(rrf_k+lex_rank) + w_sem/(rrf_k+sem_rank)

    This avoids mixing incomparable raw score scales (bm25 vs cosine).
    """
    merged: dict[_HybridKey, HybridHit] = {}

    # 1) Ingest lexical hits with ranks
    for i, hit in enumerate(lexical_hits, start=1):
        rel_path = getattr(hit, "rel_path")
        page_no = int(getattr(hit, "page_no"))
        line_no = getattr(hit, "line_no", None)

        if line_no is None:
            key = _HybridKey(str(rel_path), int(page_no), 0, 0)
        else:
            ln = int(line_no)
            key = _HybridKey(str(rel_path), int(page_no), ln, ln)

        # choose a readable text field
        text = ""
        if prefer_text == "lexical":
            text = str(getattr(hit, "snippet", "") or getattr(hit, "text", "") or "")
        else:
            text = str(getattr(hit, "text", "") or getattr(hit, "snippet", "") or "")

        bm25_score = None
        try:
            bm25_score = float(getattr(hit, "score"))
        except Exception:
            bm25_score = None

        merged[key] = HybridHit(
            key=key,
            text=text,
            lex_rank=int(i),
            lex_score=bm25_score,
            sem_rank=None,
            sem_score=None,
            score=0.0,
        )

    # 2) Ingest semantic hits with ranks
    for i, (
        sem_score,
        _chunk_id,
        rel_path,
        page_no,
        start_ln,
        end_ln,
        text,
    ) in enumerate(semantic_hits, start=1):
        key = _HybridKey(str(rel_path), int(page_no), int(start_ln), int(end_ln))

        if key in merged:
            prev = merged[key]
            # Choose which text to keep
            keep_text = prev.text
            if not keep_text or prefer_text == "semantic":
                keep_text = str(text or "") or keep_text

            merged[key] = HybridHit(
                key=key,
                text=keep_text,
                lex_rank=prev.lex_rank,
                lex_score=prev.lex_score,
                sem_rank=int(i),
                sem_score=float(sem_score),
                score=0.0,
            )
        else:
            merged[key] = HybridHit(
                key=key,
                text=str(text or ""),
                lex_rank=None,
                lex_score=None,
                sem_rank=int(i),
                sem_score=float(sem_score),
                score=0.0,
            )

    # 3) Compute RRF scores
    out: list[HybridHit] = []
    kk = max(1, int(rrf_k))
    for h in merged.values():
        s = 0.0
        s += _rrf_score(h.lex_rank, k=kk, weight=float(w_lex))
        s += _rrf_score(h.sem_rank, k=kk, weight=float(w_sem))
        out.append(
            HybridHit(
                key=h.key,
                text=h.text,
                lex_rank=h.lex_rank,
                lex_score=h.lex_score,
                sem_rank=h.sem_rank,
                sem_score=h.sem_score,
                score=float(s),
            )
        )

    out.sort(key=lambda x: x.score, reverse=True)
    return out[: max(0, int(limit))]


def _print_hybrid_hit(idx: int, hit) -> None:
    """
    Print a hybrid hit.

    Note:
      We intentionally keep this function loosely typed because hybrid hits can come from:
        - local CLI prototypes, or
        - the library module `src.index.hybrid`

      Both expose a compatible shape: `.key` with rel_path/page_no/start_line_no/end_line_no,
      plus optional lex/sem ranks and a final `.score`.
    """
    key = getattr(hit, "key", None)
    rel_path = getattr(key, "rel_path", "?")
    page_no = getattr(key, "page_no", "?")
    start_ln = getattr(key, "start_line_no", "?")
    end_ln = getattr(key, "end_line_no", "?")

    where = f"{rel_path}:{page_no}:{start_ln}-{end_ln}"

    parts = []
    lex_rank = getattr(hit, "lex_rank", None)
    sem_rank = getattr(hit, "sem_rank", None)
    if lex_rank is not None:
        parts.append(f"lex_rank={lex_rank}")
    if sem_rank is not None:
        parts.append(f"sem_rank={sem_rank}")
    meta = "  ".join(parts) if parts else "no-signals"

    score = float(getattr(hit, "score", 0.0))
    text = str(getattr(hit, "text", "") or "")

    print(f"{idx:>2}. score={score:.6f}  {meta}  {where}")
    for line in text.split("\n"):
        if line.strip():
            print(f"    {line}")
    print()


def _print_plan(plan) -> None:
    print("Query plan")
    print("---------")
    print(f"raw:        {plan.raw!r}")

    normalized_before = getattr(plan, "normalized_before_typos", None)
    fts_before_typos = getattr(plan, "fts_before_typos", None)
    fts_before_expansion = getattr(plan, "fts_before_expansion", None)

    corrections = getattr(plan, "typo_corrections", ()) or ()
    syn_expansions = getattr(plan, "synonym_expansions", ()) or ()
    morph_expansions = getattr(plan, "morphology_expansions", ()) or ()

    if normalized_before:
        print(f"normalized(before_typos): {normalized_before!r}")
    print(f"normalized:              {plan.normalized!r}")

    print(f"mode:                    {plan.mode} (advanced={bool(plan.is_advanced)})")

    if fts_before_typos:
        print(f"fts(before_typos):        {fts_before_typos!r}")
    if fts_before_expansion:
        print(f"fts(before_expansion):    {fts_before_expansion!r}")

    if fts_before_typos or fts_before_expansion:
        print(f"fts:                      {plan.fts!r}")
    else:
        print(f"fts:                     {plan.fts!r}")

    if corrections:
        print("typo_corrections:")
        for orig, corr, score, method, edits in corrections:
            edits_s = f", edits={edits}" if edits is not None else ""
            print(
                f"  - {orig!r} -> {corr!r}  score={float(score):.1f}  method={method}{edits_s}"
            )

    if syn_expansions:
        print("synonym_expansions:")
        for token, added_terms, source in syn_expansions:
            print(f"  - {token!r} + {list(added_terms)!r}  source={source}")

    if morph_expansions:
        print("morphology_expansions:")
        for token, added_terms, method in morph_expansions:
            print(f"  - {token!r} + {list(added_terms)!r}  method={method}")

    print()


def _print_line_hit(
    conn,
    idx: int,
    hit,
    *,
    show_snippet: bool,
    show_context: int,
) -> None:
    where = f"{hit.rel_path}:{hit.page_no}:{hit.line_no}"
    ocr = "ocr" if hit.is_ocr else "text"
    print(f"{idx:>2}. score={hit.score:.4f}  [{ocr}]  {where}")
    if show_snippet:
        print(f"    {hit.snippet}")
    else:
        print(f"    {hit.text}")

    if show_context > 0:
        ctx = fetch_line_context(
            conn,
            rel_path=hit.rel_path,
            page_no=hit.page_no,
            line_no=hit.line_no,
            before=int(show_context),
            after=int(show_context),
        )
        if ctx:
            print("    context:")
            for ln, txt in ctx:
                mark = ">" if ln == hit.line_no else " "
                print(f"    {mark} {ln:>4}: {txt}")

    print()


def _print_page_hit(idx: int, hit: PageHit) -> None:
    where = f"{hit.rel_path}:{hit.page_no}"
    ocr = "ocr" if hit.is_ocr else "text"
    print(
        f"{idx:>2}. score={hit.score:.4f}  [{ocr}]  {where}  (text_len={hit.text_len})"
    )
    print(f"    {hit.snippet}")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step C (Query-time, MVP): search indexed PDFs via SQLite FTS5 and BM25."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search query. If omitted, read query text from stdin.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path (default: <project_root>/data/index.sqlite)",
    )
    parser.add_argument(
        "--in",
        dest="target",
        choices=["lines", "pages"],
        default="lines",
        help="Search target: lines (precise quotes) or pages (broader context). Default: lines",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "simple", "advanced"],
        default="auto",
        help="Query mode: auto (detect advanced), simple (AND tokens), advanced (pass-through). Default: auto",
    )
    parser.add_argument(
        "--typos",
        action="store_true",
        help="Enable conservative typo correction for ordinary RU/EN words (letters-only). Default: off",
    )
    parser.add_argument(
        "--typos-mode",
        choices=["fallback", "always", "off"],
        default="fallback",
        help="Typo correction mode: fallback (only if 0 results), always, off. Default: fallback",
    )
    parser.add_argument(
        "--typos-min-len",
        type=int,
        default=4,
        help="Minimum token length eligible for typo correction. Default: 4",
    )
    parser.add_argument(
        "--typos-min-score",
        type=float,
        default=85.0,
        help="Minimum similarity score (0..100) to accept a correction. Default: 85.0",
    )
    parser.add_argument(
        "--typos-max-edits",
        type=int,
        default=2,
        help="Maximum Levenshtein edits allowed for a correction (safety guard). Default: 2",
    )
    parser.add_argument(
        "--typos-min-freq",
        type=int,
        default=2,
        help="Minimum corpus term frequency to include in lexicon. Default: 2",
    )
    parser.add_argument(
        "--typos-lexicon-limit",
        type=int,
        default=None,
        help="Optional cap on lexicon size loaded into memory. Default: no limit",
    )
    parser.add_argument(
        "--typos-force-rebuild",
        action="store_true",
        help="Force rebuild of corpus lexicon table before searching (debug).",
    )

    parser.add_argument(
        "--synonyms",
        action="store_true",
        help="Enable RU/EN synonyms expansion for ordinary words (Query-time step 3). Default: off",
    )
    parser.add_argument(
        "--synonyms-mode",
        choices=["always", "fallback", "off"],
        default="always",
        help="Synonyms expansion mode: always, fallback (only if 0 results), off. Default: always",
    )
    parser.add_argument(
        "--synonyms-path",
        default=None,
        help="Path to synonyms JSON (default: <project_root>/data/synonyms.json)",
    )
    parser.add_argument(
        "--synonyms-min-len",
        type=int,
        default=4,
        help="Minimum token length eligible for synonyms expansion. Default: 4",
    )
    parser.add_argument(
        "--synonyms-max-per-term",
        type=int,
        default=6,
        help="Max synonyms per token/phrase. Default: 6",
    )
    parser.add_argument(
        "--no-synonym-phrases",
        action="store_true",
        help="Disable multi-word phrase synonyms (use only single-token synonyms).",
    )

    parser.add_argument(
        "--morphology",
        action="store_true",
        help="Enable conservative morphology expansion (RU via pymorphy3 if installed; EN heuristics). Default: off",
    )
    parser.add_argument(
        "--morphology-mode",
        choices=["always", "fallback", "off"],
        default="always",
        help="Morphology expansion mode: always, fallback (only if 0 results), off. Default: always",
    )
    parser.add_argument(
        "--morphology-min-len",
        type=int,
        default=4,
        help="Minimum token length eligible for morphology expansion. Default: 4",
    )
    parser.add_argument(
        "--morphology-ru-max-variants",
        type=int,
        default=10,
        help="Max Russian morphology variants per token. Default: 10",
    )
    parser.add_argument(
        "--morphology-en-max-variants",
        type=int,
        default=6,
        help="Max English morphology variants per token. Default: 6",
    )
    parser.add_argument(
        "--no-morphology-ru",
        action="store_true",
        help="Disable Russian morphology expansion even if --morphology is enabled.",
    )
    parser.add_argument(
        "--no-morphology-en",
        action="store_true",
        help="Disable English morphology expansion even if --morphology is enabled.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max results to return. Default: 20",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Result offset for pagination. Default: 0",
    )
    parser.add_argument(
        "--snippet-tokens",
        type=int,
        default=18,
        help="Max snippet tokens. Default: 18",
    )
    parser.add_argument(
        "--snippet-start",
        default="[",
        help="Snippet highlight start marker. Default: [",
    )
    parser.add_argument(
        "--snippet-end",
        default="]",
        help="Snippet highlight end marker. Default: ]",
    )
    parser.add_argument(
        "--snippet-ellipsis",
        default="…",
        help="Snippet ellipsis. Default: …",
    )
    parser.add_argument(
        "--no-snippet",
        action="store_true",
        help="For line hits, print the full stored line text instead of an FTS snippet.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="For line hits, print +/- N surrounding lines from the same page. Default: 0",
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Do not print query plan (raw/normalized/fts).",
    )

    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use embeddings-based semantic retrieval over chunks (requires Step D). Default: off",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Hybrid retrieval + reranking (Step 6): combine lexical (BM25/FTS) and semantic (embeddings) using RRF. Default: off",
    )
    parser.add_argument(
        "--hybrid-limit",
        type=int,
        default=20,
        help="Final number of hybrid results to print. Default: 20",
    )
    parser.add_argument(
        "--hybrid-lex-k",
        type=int,
        default=20,
        help="Number of lexical candidates (BM25) to pull before hybrid reranking. Default: 20",
    )
    parser.add_argument(
        "--hybrid-sem-k",
        type=int,
        default=20,
        help="Number of semantic candidates (embeddings) to pull before hybrid reranking. Default: 20",
    )
    parser.add_argument(
        "--hybrid-rrf-k",
        type=int,
        default=60,
        help="RRF constant k (bigger reduces impact of top ranks). Default: 60",
    )
    parser.add_argument(
        "--hybrid-w-lex",
        type=float,
        default=1.0,
        help="Weight for lexical RRF component. Default: 1.0",
    )
    parser.add_argument(
        "--hybrid-w-sem",
        type=float,
        default=1.0,
        help="Weight for semantic RRF component. Default: 1.0",
    )
    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=10,
        help="Number of semantic results to return. Default: 10",
    )
    parser.add_argument(
        "--semantic-min-score",
        type=float,
        default=-1.0,
        help="Minimum cosine similarity score threshold. Default: -1.0",
    )
    parser.add_argument(
        "--semantic-rel-path",
        default=None,
        help="Optional filter: only search within a single document rel_path.",
    )
    parser.add_argument(
        "--semantic-embedder",
        choices=["auto", "ollama", "sentence_transformers", "hash"],
        default="auto",
        help="Embedding backend for semantic search. Default: auto (ollama -> sentence_transformers -> hash).",
    )
    parser.add_argument(
        "--semantic-ollama-model",
        default="mxbai-embed-large",
        help="Ollama embedding model name (semantic mode).",
    )
    parser.add_argument(
        "--semantic-ollama-base-url",
        default="http://localhost:11434",
        help="Ollama base URL (semantic mode).",
    )
    parser.add_argument(
        "--semantic-ollama-timeout-s",
        type=int,
        default=60,
        help="Ollama timeout seconds (semantic mode).",
    )
    parser.add_argument(
        "--semantic-hash-dim",
        type=int,
        default=384,
        help="Hash embedder vector dim for offline semantic smoke-tests (semantic mode).",
    )
    parser.add_argument(
        "--semantic-hash-seed",
        default="older.index.hash_embedder.v1",
        help="Hash embedder seed (semantic mode).",
    )
    parser.add_argument(
        "--semantic-st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model (semantic mode).",
    )
    parser.add_argument(
        "--semantic-st-device",
        default=None,
        help="Sentence-transformers device (semantic mode).",
    )
    parser.add_argument(
        "--semantic-st-batch-size",
        type=int,
        default=32,
        help="Sentence-transformers batch size (semantic mode).",
    )

    args = parser.parse_args(argv)

    user_query = _read_query_from_args_or_stdin(args.query)
    if not user_query:
        print(
            "Empty query. Provide a query argument or pipe query text via stdin.",
            file=sys.stderr,
        )
        return 2

    db_path = Path(args.db) if args.db else (_PROJECT_ROOT / "data" / "index.sqlite")
    if not db_path.exists():
        print(
            f"DB not found: {db_path}\n"
            "Run index-time first:\n"
            "  python scripts/step_a_scan_corpus.py\n"
            "  python scripts/step_b_index_pages.py",
            file=sys.stderr,
        )
        return 2

    opts = SearchOptions(
        target=str(args.target),  # type: ignore[arg-type]
        limit=int(args.limit),
        offset=int(args.offset),
        snippet_start=str(args.snippet_start),
        snippet_end=str(args.snippet_end),
        snippet_ellipsis=str(args.snippet_ellipsis),
        snippet_tokens=int(args.snippet_tokens),
        query_mode=str(args.mode),  # type: ignore[arg-type]
        typos_enabled=bool(args.typos),
        typos_mode=str(args.typos_mode),
        typos_min_token_len=int(args.typos_min_len),
        typos_min_score=float(args.typos_min_score),
        typos_max_edits=int(args.typos_max_edits),
        typos_min_freq=int(args.typos_min_freq),
        typos_lexicon_limit=args.typos_lexicon_limit,
        typos_force_rebuild=bool(args.typos_force_rebuild),
        synonyms_enabled=bool(args.synonyms),
        synonyms_mode=str(args.synonyms_mode),
        synonyms_path=args.synonyms_path,
        synonyms_min_token_len=int(args.synonyms_min_len),
        synonyms_max_per_term=int(args.synonyms_max_per_term),
        synonyms_include_phrases=not bool(args.no_synonym_phrases),
        morphology_enabled=bool(args.morphology),
        morphology_mode=str(args.morphology_mode),
        morphology_min_token_len=int(args.morphology_min_len),
        morphology_ru_enabled=not bool(args.no_morphology_ru),
        morphology_en_enabled=not bool(args.no_morphology_en),
        morphology_ru_max_variants=int(args.morphology_ru_max_variants),
        morphology_en_max_variants=int(args.morphology_en_max_variants),
    )

    conn = connect(db_path)
    try:
        if args.hybrid:
            # Hybrid retrieval + reranking (Step 6):
            # - lexical retrieval via src.index.search (BM25/FTS5) with optional typos/synonyms/morphology
            # - semantic retrieval via src.index.hybrid over chunk embeddings (requires Step D)
            # - fusion via Reciprocal Rank Fusion (RRF)
            lex_k = int(args.hybrid_lex_k)
            sem_k = int(args.hybrid_sem_k)
            final_k = int(args.hybrid_limit)

            # Prepare optional semantic embedder. If it fails, we still run lexical-only hybrid.
            embedder = None
            semantic_error: Optional[str] = None
            try:
                preferred = []
                if args.semantic_embedder == "auto":
                    preferred = ["ollama", "sentence_transformers", "hash"]
                else:
                    preferred = [str(args.semantic_embedder)]

                embedder = create_embedder(
                    preferred=preferred,  # type: ignore[arg-type]
                    # ollama
                    ollama_model=str(args.semantic_ollama_model),
                    ollama_base_url=str(args.semantic_ollama_base_url),
                    ollama_timeout_s=int(args.semantic_ollama_timeout_s),
                    # sentence-transformers
                    st_model=str(args.semantic_st_model),
                    st_device=str(args.semantic_st_device)
                    if args.semantic_st_device
                    else None,
                    st_batch_size=int(args.semantic_st_batch_size),
                    # hash fallback
                    hash_dim=int(args.semantic_hash_dim),
                    hash_seed=str(args.semantic_hash_seed),
                    normalized=True,
                )
            except EmbeddingsError as e:
                semantic_error = f"failed to create embedder: {e}"
                embedder = None
            except Exception as e:
                semantic_error = f"{type(e).__name__}: {e}"
                embedder = None

            # Run hybrid search + RRF reranking using the shared module
            hyb_plan, hyb_hits = hybrid_search_rrf(
                conn,
                user_query,
                lexical_options=SearchOptions(
                    **{**opts.__dict__, "limit": int(lex_k), "offset": 0}
                ),
                embedder=embedder,
                options=HybridOptions(
                    lexical_k=int(lex_k),
                    semantic_k=int(sem_k),
                    final_k=int(final_k),
                    semantic_rel_path=str(args.semantic_rel_path)
                    if args.semantic_rel_path
                    else None,
                    semantic_min_score=float(args.semantic_min_score),
                    rrf_k=int(args.hybrid_rrf_k),
                    w_lex=float(args.hybrid_w_lex),
                    w_sem=float(args.hybrid_w_sem),
                    prefer_text="lexical",
                ),
            )

            if not args.no_plan:
                print("Hybrid plan")
                print("-----------")
                _print_plan(hyb_plan.lexical_plan)
                print(
                    f"hybrid:     lex_k={hyb_plan.lexical_k}, sem_k={hyb_plan.semantic_k}, final_k={hyb_plan.final_k}, "
                    f"rrf_k={hyb_plan.rrf_k}, w_lex={hyb_plan.w_lex}, w_sem={hyb_plan.w_sem}"
                )
                if semantic_error:
                    print(f"semantic:   disabled (reason: {semantic_error})")
                elif not hyb_plan.semantic_enabled:
                    reason = hyb_plan.semantic_error or "not available"
                    print(f"semantic:   disabled (reason: {reason})")
                else:
                    print(
                        f"semantic:   enabled (model_label={hyb_plan.semantic_model_label!r})"
                    )
                print()

            if not hyb_hits:
                print("No results.")
                return 0

            for i, h in enumerate(hyb_hits, start=1):
                _print_hybrid_hit(i, h)

            return 0

        if args.semantic:
            # Semantic retrieval mode (Step 5). Requires that you ran:
            #   python scripts/step_d_embed_chunks.py
            # to populate chunks + chunk_embeddings.
            preferred = []
            if args.semantic_embedder == "auto":
                preferred = ["ollama", "sentence_transformers", "hash"]
            else:
                preferred = [str(args.semantic_embedder)]

            try:
                embedder = create_embedder(
                    preferred=preferred,  # type: ignore[arg-type]
                    # ollama
                    ollama_model=str(args.semantic_ollama_model),
                    ollama_base_url=str(args.semantic_ollama_base_url),
                    ollama_timeout_s=int(args.semantic_ollama_timeout_s),
                    # sentence-transformers
                    st_model=str(args.semantic_st_model),
                    st_device=str(args.semantic_st_device)
                    if args.semantic_st_device
                    else None,
                    st_batch_size=int(args.semantic_st_batch_size),
                    # hash fallback
                    hash_dim=int(args.semantic_hash_dim),
                    hash_seed=str(args.semantic_hash_seed),
                    normalized=True,
                )
            except EmbeddingsError as e:
                print(
                    f"Semantic error: failed to create embedder: {e}", file=sys.stderr
                )
                return 2

            model_label = _semantic_model_label(embedder)

            # We reuse the same query text; previous steps (typos/synonyms) are lexical-focused.
            # Semantic retrieval typically works best on the user's original intent.
            q = user_query

            if not args.no_plan:
                print("Semantic plan")
                print("-------------")
                print(
                    f"embedder:   {embedder.info().backend}  model_label={model_label!r}  dim={embedder.info().dim}"
                )
                print(f"query:      {q!r}")
                if args.semantic_rel_path:
                    print(f"rel_path:   {args.semantic_rel_path!r}")
                print()

            try:
                hits = _semantic_search_sql(
                    conn,
                    embedder=embedder,
                    model_label=str(model_label),
                    query=str(q),
                    top_k=int(args.semantic_top_k),
                    min_score=float(args.semantic_min_score),
                    rel_path=str(args.semantic_rel_path)
                    if args.semantic_rel_path
                    else None,
                )
            except Exception as e:
                print(f"Semantic error: {type(e).__name__}: {e}", file=sys.stderr)
                return 2

            if not hits:
                print("No results.")
                return 0

            for i, (
                score,
                chunk_id,
                rel_path,
                page_no,
                start_ln,
                end_ln,
                text,
            ) in enumerate(hits, start=1):
                where = f"{rel_path}:{page_no}:{start_ln}-{end_ln}"
                print(
                    f"{i:>2}. score={float(score):.4f}  chunk_id={int(chunk_id)}  {where}"
                )
                # Print the chunk text (already stored as a small window)
                for line in (text or "").split("\n"):
                    if line.strip():
                        print(f"    {line}")
                print()

            return 0

        # Default: lexical retrieval mode (FTS5 + BM25), with optional typos/synonyms/morphology.
        try:
            plan, hits = search(conn, user_query, options=opts)
        except SearchError as e:
            print(f"Search error: {e}", file=sys.stderr)
            return 2

        if not args.no_plan:
            _print_plan(plan)

        if not hits:
            print("No results.")
            return 0

        if args.target == "lines":
            for i, hit in enumerate(hits, start=1):
                _print_line_hit(
                    conn,
                    i,
                    hit,
                    show_snippet=not bool(args.no_snippet),
                    show_context=int(args.context),
                )
        else:
            for i, hit in enumerate(hits, start=1):
                _print_page_hit(i, cast(PageHit, hit))

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
