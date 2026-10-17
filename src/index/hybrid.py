from __future__ import annotations

"""
Hybrid retrieval + reranking (Query-time Step 6): combine lexical BM25 (FTS5) with
semantic embeddings (chunk vectors) without using an LLM.

This module is designed to fit cleanly into the existing project structure:

- Lexical retrieval (Step 4) lives in:  src.index.search
- Semantic artifacts (Step 5) live in SQLite tables:
    - chunks
    - chunk_embeddings
  and are built by: scripts/step_d_embed_chunks.py
- Embedding backends live in: src.index.embeddings

What this module provides
-------------------------
1) Semantic retrieval (MVP) by scanning chunk embeddings in SQLite (brute-force cosine).
   - This is intentionally dependency-light and correct, not necessarily fast for huge corpora.
   - You can swap it with an ANN index later (FAISS/HNSW) without changing the output format.

2) Hybrid reranking using Reciprocal Rank Fusion (RRF).
   - RRF avoids mixing incomparable raw score scales (bm25 vs cosine).
   - It is robust and easy to tune with weights.

3) An explainable result model (HybridHit) that carries:
   - stable citation key (rel_path, page_no, line range)
   - text
   - ranks and raw scores from each channel (optional)
   - final fused score

Recommended usage pattern
-------------------------
- Run index-time:
    Step A -> Step B -> Step D (embed chunks)
- At query-time:
    - run lexical search (optionally with typos/synonyms/morphology)
    - run semantic retrieval over chunks
    - fuse and rerank with RRF
    - print top-N with sources and text snippets

Notes
-----
- Semantic retrieval depends on chunk_embeddings.model matching the same stable model label
  used at indexing time. Use embeddings.stable_model_label_for_embedder(embedder).
- By default, this module treats embeddings as cosine-normalized vectors if the embedder
  reports `info.normalized=True`. If not normalized, it computes cosine via norms.

This module does NOT implement:
- LLM prompting
- advanced boolean query parsing beyond what FTS5 already supports (handled in src.index.search)
"""

import math
import sqlite3
from array import array
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from . import search as lexical
from .embeddings import Embedder, EmbeddingsError, stable_model_label_for_embedder

# -----------------------------
# Models
# -----------------------------


@dataclass(frozen=True)
class HybridKey:
    """
    Stable citation key used for merging/dedup and for user-facing references.
    """

    rel_path: str
    page_no: int
    start_line_no: int
    end_line_no: int


@dataclass(frozen=True)
class SemanticCandidate:
    key: HybridKey
    chunk_id: int
    score: float  # cosine similarity (higher is better)
    text: str


@dataclass(frozen=True)
class LexicalCandidate:
    key: HybridKey
    rank: int  # 1-based rank in lexical list
    score: Optional[
        float
    ]  # bm25 from FTS (lower is better); not used in fusion by default
    text: str


@dataclass(frozen=True)
class HybridHit:
    key: HybridKey
    text: str

    # Signals from each channel (optional)
    lex_rank: Optional[int] = None
    lex_score: Optional[float] = None  # bm25 (debug/inspection only)

    sem_rank: Optional[int] = None
    sem_score: Optional[float] = None  # cosine similarity (debug/inspection only)

    # Final fused score (higher is better)
    score: float = 0.0


@dataclass(frozen=True)
class HybridPlan:
    """
    Explainability container for a hybrid run.
    """

    user_query: str

    lexical_plan: lexical.QueryPlan
    lexical_k: int

    semantic_enabled: bool
    semantic_model_label: Optional[str]
    semantic_k: int
    semantic_error: Optional[str]

    fusion: str  # e.g. "rrf"
    rrf_k: int
    w_lex: float
    w_sem: float

    final_k: int


@dataclass(frozen=True)
class HybridOptions:
    """
    High-level hybrid tuning knobs.
    """

    lexical_k: int = 20
    semantic_k: int = 20
    final_k: int = 20

    # Optional filter for semantic retrieval (single document)
    semantic_rel_path: Optional[str] = None

    # Semantic score cutoff (cosine); keep as -1.0 to disable cutoff.
    semantic_min_score: float = -1.0

    # RRF parameters
    rrf_k: int = 60
    w_lex: float = 1.0
    w_sem: float = 1.0

    # Which text to keep when both channels hit the same key
    prefer_text: str = "lexical"  # "lexical" | "semantic"


class HybridError(RuntimeError):
    """User-facing error for hybrid retrieval/reranking failures."""


# -----------------------------
# Helpers: vector decoding & cosine
# -----------------------------


def _decode_f32(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b, strict=False):
        s += float(x) * float(y)
    return s


def _l2_norm(vec: Sequence[float]) -> float:
    s2 = 0.0
    for x in vec:
        s2 += float(x) * float(x)
    return math.sqrt(s2) if s2 > 0.0 else 0.0


# -----------------------------
# Lexical candidates
# -----------------------------


def _lexical_candidates_from_hits(
    hits: Sequence[object], *, prefer_text: str = "lexical"
) -> list[LexicalCandidate]:
    out: list[LexicalCandidate] = []

    for i, hit in enumerate(hits, start=1):
        rel_path = str(getattr(hit, "rel_path"))
        page_no = int(getattr(hit, "page_no"))

        line_no = getattr(hit, "line_no", None)
        if line_no is None:
            key = HybridKey(
                rel_path=rel_path, page_no=page_no, start_line_no=0, end_line_no=0
            )
        else:
            ln = int(line_no)
            key = HybridKey(
                rel_path=rel_path, page_no=page_no, start_line_no=ln, end_line_no=ln
            )

        # Choose a human-readable text
        snippet = str(getattr(hit, "snippet", "") or "")
        text = str(getattr(hit, "text", "") or "")
        chosen = snippet if (prefer_text == "lexical") else text
        if not chosen:
            chosen = text or snippet

        bm25_score: Optional[float]
        try:
            bm25_score = float(getattr(hit, "score"))
        except Exception:
            bm25_score = None

        out.append(LexicalCandidate(key=key, rank=i, score=bm25_score, text=chosen))

    return out


# -----------------------------
# Semantic retrieval (SQLite brute-force)
# -----------------------------


def semantic_retrieve_sqlite(
    conn: sqlite3.Connection,
    *,
    embedder: Embedder,
    query: str,
    top_k: int,
    min_score: float = -1.0,
    rel_path: Optional[str] = None,
) -> tuple[str, list[SemanticCandidate]]:
    """
    Retrieve semantic candidates by scanning embeddings in SQLite.

    Requires:
      - chunks table
      - chunk_embeddings table populated for stable_model_label_for_embedder(embedder)

    Returns:
      (model_label, candidates_sorted_desc_by_score)
    """
    if int(top_k) <= 0:
        return stable_model_label_for_embedder(embedder), []

    info = embedder.info()
    model_label = stable_model_label_for_embedder(embedder)

    q = (query or "").strip()
    if not q:
        return model_label, []

    try:
        q_vec = embedder.embed_query(q)
    except EmbeddingsError as e:
        raise HybridError(f"Failed to embed query for semantic retrieval: {e}") from e

    assume_normalized = bool(info.normalized)
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
        JOIN chunks c      ON c.chunk_id = e.chunk_id
        JOIN pages p       ON p.page_id  = c.page_id
        JOIN documents d   ON d.doc_id   = p.doc_id
        {where}
        """,
        tuple(params),
    ).fetchall()

    # Keep a small top-k list without sorting the whole corpus.
    # We store as list and replace the minimum when necessary.
    best: list[SemanticCandidate] = []
    best_min: float = float("-inf")

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

        cand = SemanticCandidate(
            key=HybridKey(
                rel_path=rp, page_no=page_no, start_line_no=start_ln, end_line_no=end_ln
            ),
            chunk_id=chunk_id,
            score=float(score),
            text=text,
        )

        if len(best) < int(top_k):
            best.append(cand)
            if len(best) == 1:
                best_min = cand.score
            else:
                best_min = min(best_min, cand.score)
        else:
            # Replace the current minimum if this score is better.
            if cand.score <= best_min:
                continue
            # Find index of current min (top_k is small; O(k) is fine).
            min_idx = 0
            min_val = best[0].score
            for j in range(1, len(best)):
                if best[j].score < min_val:
                    min_val = best[j].score
                    min_idx = j
            best[min_idx] = cand
            # Recompute min
            best_min = min(x.score for x in best)

    best.sort(key=lambda x: x.score, reverse=True)
    return model_label, best


# -----------------------------
# Fusion / reranking (RRF)
# -----------------------------


def _rrf_component(rank: Optional[int], *, k: int, weight: float) -> float:
    """
    Reciprocal Rank Fusion component:
      weight / (k + rank)
    rank is 1-based. If rank is None -> 0.
    """
    if rank is None:
        return 0.0
    r = int(rank)
    if r <= 0:
        return 0.0
    return float(weight) / float(int(k) + r)


def rerank_rrf(
    *,
    lexical: Sequence[LexicalCandidate],
    semantic: Sequence[SemanticCandidate],
    rrf_k: int = 60,
    w_lex: float = 1.0,
    w_sem: float = 1.0,
    limit: int = 20,
    prefer_text: str = "lexical",
) -> list[HybridHit]:
    """
    Merge & rerank lexical and semantic candidates using RRF.

    - lexical candidate ranks come from their list order.
    - semantic candidate ranks come from their list order.

    Raw bm25/cosine scores are stored for debugging only; the final score is RRF-based.
    """
    kk = max(1, int(rrf_k))
    out_limit = max(0, int(limit))

    merged: dict[HybridKey, HybridHit] = {}

    # ingest lexical
    for c in lexical:
        merged[c.key] = HybridHit(
            key=c.key,
            text=c.text,
            lex_rank=int(c.rank),
            lex_score=c.score,
            sem_rank=None,
            sem_score=None,
            score=0.0,
        )

    # ingest semantic
    for i, c in enumerate(semantic, start=1):
        if c.key in merged:
            prev = merged[c.key]
            chosen_text = prev.text
            if not chosen_text or prefer_text == "semantic":
                chosen_text = c.text or chosen_text
            merged[c.key] = HybridHit(
                key=prev.key,
                text=chosen_text,
                lex_rank=prev.lex_rank,
                lex_score=prev.lex_score,
                sem_rank=int(i),
                sem_score=float(c.score),
                score=0.0,
            )
        else:
            merged[c.key] = HybridHit(
                key=c.key,
                text=c.text,
                lex_rank=None,
                lex_score=None,
                sem_rank=int(i),
                sem_score=float(c.score),
                score=0.0,
            )

    fused: list[HybridHit] = []
    for h in merged.values():
        s = 0.0
        s += _rrf_component(h.lex_rank, k=kk, weight=float(w_lex))
        s += _rrf_component(h.sem_rank, k=kk, weight=float(w_sem))
        fused.append(
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

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:out_limit]


# -----------------------------
# High-level hybrid API
# -----------------------------


def hybrid_search_rrf(
    conn: sqlite3.Connection,
    user_query: str,
    *,
    lexical_options: Optional[lexical.SearchOptions] = None,
    embedder: Optional[Embedder] = None,
    options: Optional[HybridOptions] = None,
) -> tuple[HybridPlan, list[HybridHit]]:
    """
    Perform hybrid retrieval and reranking using RRF.

    - Lexical channel: always runs (FTS5/BM25), using src.index.search.search().
    - Semantic channel: runs only if `embedder` is provided and chunk embeddings exist.

    Returns:
      (HybridPlan, HybridHits)
    """
    opts = options or HybridOptions()

    # 1) Lexical retrieval
    lex_opts = lexical_options or lexical.SearchOptions()
    # Override limit for candidate pool size
    lex_opts = lexical.SearchOptions(
        **{**lex_opts.__dict__, "limit": int(opts.lexical_k), "offset": 0}
    )
    lex_plan, lex_hits = lexical.search(conn, user_query, options=lex_opts)
    lex_cands = _lexical_candidates_from_hits(lex_hits, prefer_text="lexical")

    # 2) Semantic retrieval (optional)
    sem_enabled = embedder is not None
    sem_model_label: Optional[str] = None
    sem_error: Optional[str] = None
    sem_cands: list[SemanticCandidate] = []

    if sem_enabled:
        try:
            sem_model_label, sem_cands = semantic_retrieve_sqlite(
                conn,
                embedder=embedder,  # type: ignore[arg-type]
                query=user_query,
                top_k=int(opts.semantic_k),
                min_score=float(opts.semantic_min_score),
                rel_path=opts.semantic_rel_path,
            )
        except Exception as e:
            sem_error = f"{type(e).__name__}: {e}"
            sem_enabled = False
            sem_model_label = None
            sem_cands = []

    # 3) Rerank with RRF
    fused = rerank_rrf(
        lexical=lex_cands,
        semantic=sem_cands,
        rrf_k=int(opts.rrf_k),
        w_lex=float(opts.w_lex),
        w_sem=float(opts.w_sem),
        limit=int(opts.final_k),
        prefer_text=str(opts.prefer_text),
    )

    plan = HybridPlan(
        user_query=str(user_query),
        lexical_plan=lex_plan,
        lexical_k=int(opts.lexical_k),
        semantic_enabled=bool(sem_enabled and (sem_error is None)),
        semantic_model_label=sem_model_label,
        semantic_k=int(opts.semantic_k),
        semantic_error=sem_error,
        fusion="rrf",
        rrf_k=int(opts.rrf_k),
        w_lex=float(opts.w_lex),
        w_sem=float(opts.w_sem),
        final_k=int(opts.final_k),
    )

    return plan, fused


# -----------------------------
# Formatting helpers (optional)
# -----------------------------


def format_citation(key: HybridKey) -> str:
    """
    Format a stable citation string.
    """
    return f"{key.rel_path}:{key.page_no}:{key.start_line_no}-{key.end_line_no}"


__all__ = [
    "HybridKey",
    "SemanticCandidate",
    "LexicalCandidate",
    "HybridHit",
    "HybridPlan",
    "HybridOptions",
    "HybridError",
    "semantic_retrieve_sqlite",
    "rerank_rrf",
    "hybrid_search_rrf",
    "format_citation",
]
