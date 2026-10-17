from __future__ import annotations

"""
Interactive REPL application for this project.

Goal
----
Provide a simple "app-like" UX:

  python main.py
  # then type queries and commands:
  > voltage reference
  > :mode hybrid
  > :typos on
  > :index
  > :embed hash 128

This REPL is a thin orchestration layer over the existing pipeline:

Index-time:
  - Step A: scripts/step_a_scan_corpus.py
  - Step B: scripts/step_b_index_pages.py
  - Step D: scripts/step_d_embed_chunks.py   (semantic chunks + embeddings)

Query-time:
  - Lexical: src.index.search.search()       (FTS5 MATCH + BM25 + optional typos/syn/morph)
  - Hybrid:  src.index.hybrid.hybrid_search_rrf() (lexical + semantic + RRF)

Important principles
--------------------
- DO NOT modify or break existing scripts. We call them as subprocesses when needed.
- Keep state explicit and printable (future-friendly for adding LLM as ":answer").
- Default mode is lexical because it is the most robust and only requires Step B.
"""

import argparse
import json
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence

from src.index.db import connect, init_db
from src.index.embeddings import (
    EmbeddingsError,
    create_embedder,
    stable_model_label_for_embedder,
)
from src.index.hybrid import HybridOptions, hybrid_search_rrf
from src.index.llm import (
    LLMContextItem,
    LLMError,
    OllamaChat,
    OllamaChatConfig,
    answer_with_context,
)
from src.index.search import SearchError, SearchOptions, fetch_line_context, search

Mode = Literal["lexical", "semantic", "hybrid"]
Target = Literal["lines", "pages"]
SemanticBackend = Literal["auto", "ollama", "sentence_transformers", "hash"]


# -----------------------------
# Models (future-friendly for LLM)
# -----------------------------


@dataclass(frozen=True)
class Citation:
    rel_path: str
    page_no: int
    start_line_no: int = 0
    end_line_no: int = 0

    def format(self) -> str:
        if self.start_line_no and self.end_line_no:
            return f"{self.rel_path}:{self.page_no}:{self.start_line_no}-{self.end_line_no}"
        return f"{self.rel_path}:{self.page_no}"


@dataclass(frozen=True)
class ContextItem:
    """
    A chunk of text with a stable source reference.

    This is the exact shape you'd later pass to an LLM.
    """

    citation: Citation
    text: str
    channel: str  # "lexical" | "semantic" | "hybrid"
    score: Optional[float] = None


@dataclass
class AppState:
    # Paths
    project_root: Path
    db_path: Path

    # Startup behavior
    # If enabled, the REPL will check DB readiness and offer to run :index / :embed.
    startup_prompts: bool = True

    # Query-time defaults
    mode: Mode = "lexical"
    target: Target = "lines"
    limit: int = 10
    context_lines: int = 2
    show_plan: bool = True

    # Step (2) typos
    typos: bool = False

    # Step (3) synonyms/morphology
    synonyms: bool = False
    morphology: bool = False

    # Advanced/safe query mode
    query_mode: Literal["auto", "simple", "advanced"] = "auto"

    # Semantic settings (Step 5) used by semantic/hybrid modes
    semantic_backend: SemanticBackend = "hash"
    semantic_hash_dim: int = 128
    semantic_hash_seed: str = "older.index.hash_embedder.v1"

    semantic_ollama_model: str = "mxbai-embed-large"
    semantic_ollama_base_url: str = "http://localhost:11434"
    semantic_ollama_timeout_s: int = 60

    semantic_st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_st_device: Optional[str] = None
    semantic_st_batch_size: int = 32

    # Hybrid / RRF tuning (Step 6)
    hybrid_lex_k: int = 20
    hybrid_sem_k: int = 20
    hybrid_rrf_k: int = 60
    hybrid_w_lex: float = 1.0
    hybrid_w_sem: float = 1.0

    # Optional semantic filtering
    semantic_min_score: float = -1.0
    semantic_rel_path: Optional[str] = None

    # Last context (used by the LLM step / future ":answer")
    last_query: Optional[str] = None
    last_context: list[ContextItem] = field(default_factory=list)

    # Step 7: LLM (Ollama chat) settings
    # By default the LLM is ON: any plain query runs retrieval + LLM answer.
    llm_enabled: bool = True
    llm_model: str = "t-lite-it"
    llm_base_url: str = "http://localhost:11434"
    llm_timeout_s: int = 300
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_num_predict: int = 1024
    llm_stream: bool = True
    llm_max_chunks: int = 6
    llm_max_context_chars: int = 12000
    llm_max_chars_per_item: int = 2000
    # When True, do not call LLM if retrieval returned zero context items.
    llm_skip_if_no_context: bool = True

    # Step 3.1: context expansion (only affects what is sent to LLM,
    # retrieval results in the printed plan stay exactly the same).
    # When True, each retrieved chunk is widened by +/- N lines on the same page
    # before being sent to the LLM, and chunks from the same (file, page) are merged.
    llm_expand_context: bool = True
    llm_expand_lines: int = 15
    # Larger per-item char cap used when expansion is on (so the expanded
    # paragraph is not chopped). The retrieval-mode cap remains llm_max_chars_per_item.
    llm_expand_max_chars_per_item: int = 4000

    # Step 3.3: cross-lingual support via LLM-based query translation.
    # When True and the LLM is enabled, queries containing Cyrillic characters are
    # translated to English via T-Pro before retrieval. The original query is still
    # passed to the LLM for the final answer (so the user gets a Russian reply).
    # Both forms are printed in the plan and stored in answers.jsonl.
    # An override via the ":translate <english>" command always wins over the auto
    # translation (cached per original query).
    llm_translate_query: bool = True
    llm_translate_min_cyrillic_chars: int = 2

    # When True, print a compact summary of what the LLM actually received as context
    # AFTER expansion (helps the user verify that the answer's citations point to text
    # that contains the answer). Only meaningful when llm_expand_context is on.
    llm_show_expanded: bool = True


# -----------------------------
# Utilities
# -----------------------------


def _print_block(title: str, body: str) -> None:
    print(title)
    print("-" * len(title))
    print(body.rstrip())
    print()


def _fmt_bool(v: bool) -> str:
    return "on" if v else "off"


def _safe_int(s: str, name: str) -> int:
    try:
        return int(s)
    except Exception as e:
        raise ValueError(f"{name} must be an integer, got {s!r}") from e


def _project_root_from_this_file() -> Path:
    # repl.py -> src/app/repl.py => project root is parents[2]
    return Path(__file__).resolve().parents[2]


def _default_db_path(project_root: Path) -> Path:
    return project_root / "data" / "index.sqlite"


def _run_script(project_root: Path, args: Sequence[str]) -> int:
    """
    Run a project script as a subprocess, preserving stdout/stderr.

    We use subprocess so we:
      - do not depend on internal function refactors
      - do not break existing scripts
    """
    cmd = [sys.executable, *args]
    proc = subprocess.run(cmd, cwd=str(project_root))
    return int(proc.returncode)


def _ensure_db_initialized(state: AppState) -> None:
    """
    Ensure the DB exists and has the latest schema (safe to call repeatedly).
    """
    state.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(state.db_path)
    try:
        init_db(conn)
        conn.commit()
    finally:
        conn.close()


def _prompt_yes_no(question: str, *, default_yes: bool = True) -> bool:
    """
    Interactive yes/no prompt.
    Returns True for yes, False for no.
    """
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        ans = input(f"{question} {suffix} ").strip().lower()
        if not ans:
            return bool(default_yes)
        if ans in ("y", "yes", "1", "true", "on"):
            return True
        if ans in ("n", "no", "0", "false", "off"):
            return False
        print("Please answer yes or no.")


def _db_count(conn, table: str) -> Optional[int]:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None


def _startup_readiness_check(state: AppState) -> None:
    """
    On startup, check whether the DB is ready for queries and offer to run indexing.

    - If DB is missing: offer to run :index.
    - If DB exists but has 0 pages/lines: offer to run :index.
    - If mode requires embeddings (semantic/hybrid) but no chunk embeddings exist: offer to run :embed.

    This does NOT break existing scripts; it reuses them through the same handlers.
    """
    if not state.startup_prompts:
        return

    if not state.db_path.exists():
        print(f"DB not found: {state.db_path}")
        if _prompt_yes_no("Run indexing now? (:index)", default_yes=True):
            try:
                _handle_index(state, [])
            except Exception as e:
                print(f"Startup index failed: {type(e).__name__}: {e}\n")
        return

    # DB exists: ensure schema is up-to-date before probing.
    try:
        _ensure_db_initialized(state)
    except Exception as e:
        print(f"DB init failed: {type(e).__name__}: {e}\n")
        return

    conn = connect(state.db_path)
    try:
        pages = _db_count(conn, "pages")
        lines = _db_count(conn, "lines")

        if pages == 0 or lines == 0:
            print(f"Index appears empty (pages={pages}, lines={lines}).")
            if _prompt_yes_no("Run indexing now? (:index)", default_yes=True):
                try:
                    _handle_index(state, [])
                except Exception as e:
                    print(f"Startup index failed: {type(e).__name__}: {e}\n")
            return

        # LLM readiness: if the LLM is enabled by default, give a one-line hint
        # so the user understands why their query may stall on the first run.
        if state.llm_enabled:
            try:
                client = _make_llm_client(state)
                if not client.ping():
                    print(
                        f"LLM hint: Ollama is not reachable at {state.llm_base_url}.\n"
                        f"  Start it with `ollama serve`, install the model with\n"
                        f"  `ollama pull {state.llm_model}` (or another chat model), or disable\n"
                        f"  the LLM step with :llm off.\n"
                    )
                else:
                    models = client.list_models()
                    # Ollama always reports model names with an explicit tag
                    # (e.g. "t-pro-it:latest"), but users typically configure
                    # the short form ("t-pro-it"). Normalize both sides before
                    # checking, otherwise we print a false "not installed" hint.
                    def _norm(n: str) -> str:
                        return n if ":" in n else f"{n}:latest"
                    configured = _norm(state.llm_model)
                    installed_norm = {_norm(m) for m in models}
                    if models and configured not in installed_norm:
                        # Be helpful but non-blocking: pulling can take a long time.
                        print(
                            f"LLM hint: model {state.llm_model!r} is not in Ollama yet.\n"
                            f"  Installed: {', '.join(models) or '(none)'}\n"
                            f"  Run `ollama pull {state.llm_model}` or switch with :llm model NAME.\n"
                        )
            except Exception:
                # Never block startup on LLM probing.
                pass

        # If the user starts in semantic/hybrid, but embeddings are missing, offer to build them.
        if state.mode in ("semantic", "hybrid"):
            embeds = _db_count(conn, "chunk_embeddings")
            chunks = _db_count(conn, "chunks")
            if (embeds is not None and embeds == 0) or (
                chunks is not None and chunks == 0
            ):
                print(f"No embeddings found (chunks={chunks}, embeddings={embeds}).")
                msg = f"Build embeddings now? (:embed hash {state.semantic_hash_dim})"
                if _prompt_yes_no(msg, default_yes=False):
                    try:
                        _handle_embed(state, ["hash", str(state.semantic_hash_dim)])
                    except Exception as e:
                        print(f"Startup embed failed: {type(e).__name__}: {e}\n")
            else:
                # Embeddings exist, but possibly not for the *currently selected*
                # semantic model. Warn so the user does not silently get hybrid
                # falling back to lexical-only retrieval.
                try:
                    embedder = _create_semantic_embedder(state)
                    label = stable_model_label_for_embedder(embedder)
                    row = conn.execute(
                        "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?",
                        (label,),
                    ).fetchone()
                    cnt_for_model = int(row[0]) if row else 0
                    if cnt_for_model == 0:
                        cur_model_arg = (
                            state.semantic_ollama_model
                            if state.semantic_backend in ("ollama", "auto")
                            else state.semantic_st_model
                            if state.semantic_backend == "sentence_transformers"
                            else str(state.semantic_hash_dim)
                        )
                        print(
                            f"Semantic hint: no embeddings yet for the selected model.\n"
                            f"  backend: {state.semantic_backend}   model_label: {label!r}\n"
                            f"  Run `:embed {state.semantic_backend} {cur_model_arg}` "
                            f"to compute them, or use `:sem status` to inspect what is in the DB.\n"
                        )
                except Exception:
                    # Embedder may not be available (e.g. Ollama down).
                    # We do not block startup on this.
                    pass

    finally:
        conn.close()


# -----------------------------
# Retrieval helpers (produce ContextItem lists)
# -----------------------------


def _lexical_context(state: AppState, query: str) -> tuple[object, list[ContextItem]]:
    """
    Run lexical retrieval using src.index.search.search and build context items.

    Returns:
      (plan, context_items)
    """
    opts = SearchOptions(
        target=state.target,
        limit=int(state.limit),
        offset=0,
        query_mode=state.query_mode,
        typos_enabled=bool(state.typos),
        synonyms_enabled=bool(state.synonyms),
        morphology_enabled=bool(state.morphology),
    )

    conn = connect(state.db_path)
    try:
        plan, hits = search(conn, query, options=opts)
        items: list[ContextItem] = []

        # Lines target -> show line context (window) for quote-friendly chunks
        if state.target == "lines":
            for h in hits:
                rel_path = getattr(h, "rel_path")
                page_no = int(getattr(h, "page_no"))
                line_no = int(getattr(h, "line_no"))

                # Build a window of lines around match
                ctx = fetch_line_context(
                    conn,
                    rel_path=str(rel_path),
                    page_no=int(page_no),
                    line_no=int(line_no),
                    before=int(state.context_lines),
                    after=int(state.context_lines),
                )
                if ctx:
                    text = "\n".join(t for _ln, t in ctx)
                    start_ln = int(ctx[0][0])
                    end_ln = int(ctx[-1][0])
                else:
                    # Fallback to the matched line only
                    text = str(
                        getattr(h, "text", "") or getattr(h, "snippet", "") or ""
                    )
                    start_ln = int(line_no)
                    end_ln = int(line_no)

                items.append(
                    ContextItem(
                        citation=Citation(
                            rel_path=str(rel_path),
                            page_no=int(page_no),
                            start_line_no=start_ln,
                            end_line_no=end_ln,
                        ),
                        text=text,
                        channel="lexical",
                        score=float(getattr(h, "score", 0.0)),
                    )
                )

        # Pages target -> use snippets as "chunks"
        else:
            for h in hits:
                rel_path = getattr(h, "rel_path")
                page_no = int(getattr(h, "page_no"))
                snippet = str(getattr(h, "snippet", "") or "")
                items.append(
                    ContextItem(
                        citation=Citation(rel_path=str(rel_path), page_no=int(page_no)),
                        text=snippet,
                        channel="lexical",
                        score=float(getattr(h, "score", 0.0)),
                    )
                )

        return plan, items
    finally:
        conn.close()


def _create_semantic_embedder(state: AppState):
    preferred: list[str]
    if state.semantic_backend == "auto":
        preferred = ["ollama", "sentence_transformers", "hash"]
    else:
        preferred = [str(state.semantic_backend)]

    return create_embedder(
        preferred=preferred,  # type: ignore[arg-type]
        # ollama
        ollama_model=state.semantic_ollama_model,
        ollama_base_url=state.semantic_ollama_base_url,
        ollama_timeout_s=int(state.semantic_ollama_timeout_s),
        # sentence-transformers
        st_model=state.semantic_st_model,
        st_device=state.semantic_st_device,
        st_batch_size=int(state.semantic_st_batch_size),
        # hash fallback
        hash_dim=int(state.semantic_hash_dim),
        hash_seed=state.semantic_hash_seed,
        normalized=True,
    )


def _hybrid_context(state: AppState, query: str) -> tuple[object, list[ContextItem]]:
    """
    Run hybrid retrieval + RRF reranking using src.index.hybrid.hybrid_search_rrf.

    Returns:
      (hybrid_plan, context_items)
    """
    embedder = None
    semantic_error: Optional[str] = None
    try:
        embedder = _create_semantic_embedder(state)
    except EmbeddingsError as e:
        embedder = None
        semantic_error = str(e)

    # Lexical options reused from state
    lex_opts = SearchOptions(
        target=state.target,
        limit=int(state.hybrid_lex_k),
        offset=0,
        query_mode=state.query_mode,
        typos_enabled=bool(state.typos),
        synonyms_enabled=bool(state.synonyms),
        morphology_enabled=bool(state.morphology),
    )

    conn = connect(state.db_path)
    try:
        hyb_plan, hyb_hits = hybrid_search_rrf(
            conn,
            query,
            lexical_options=lex_opts,
            embedder=embedder,
            options=HybridOptions(
                lexical_k=int(state.hybrid_lex_k),
                semantic_k=int(state.hybrid_sem_k),
                final_k=int(state.limit),
                semantic_rel_path=state.semantic_rel_path,
                semantic_min_score=float(state.semantic_min_score),
                rrf_k=int(state.hybrid_rrf_k),
                w_lex=float(state.hybrid_w_lex),
                w_sem=float(state.hybrid_w_sem),
                prefer_text="lexical",
            ),
        )
    finally:
        conn.close()

    # If semantic failed to initialize, annotate plan (for printing)
    if semantic_error and getattr(hyb_plan, "semantic_error", None) is None:
        # dataclass is frozen in module; we won't mutate it. Just print error in REPL.
        pass

    items: list[ContextItem] = []
    for h in hyb_hits:
        c = Citation(
            rel_path=str(h.key.rel_path),
            page_no=int(h.key.page_no),
            start_line_no=int(h.key.start_line_no),
            end_line_no=int(h.key.end_line_no),
        )
        items.append(
            ContextItem(
                citation=c,
                text=str(h.text or ""),
                channel="hybrid",
                score=float(h.score),
            )
        )

    return hyb_plan, items


# -----------------------------
# LLM helpers (Step 7)
# -----------------------------


# Module-level translation cache.
# Keyed by the exact original query string; value is the English translation.
# A ":translate <english>" override writes directly here too.
# In-memory only (per session) — survives between queries, not between REPL restarts.
_translation_cache: dict[str, str] = {}


def _has_enough_cyrillic(text: str, min_chars: int) -> bool:
    n = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return n >= max(1, int(min_chars))


def _maybe_translate_query(state: AppState, original: str) -> Optional[str]:
    """
    Step 3.3: translate a Russian query to English via the chat LLM.

    Returns the translated query, or None if translation was not performed
    (translation disabled, LLM disabled or unreachable, query is already
    English-like, weird LLM output, etc.).

    Conservative rules:
      - Need llm_translate_query AND llm_enabled to be True.
      - Need at least N Cyrillic characters in the query.
      - LLM must be reachable (no silent retries).
      - Strip surrounding quotes / common "Translation: " prefixes the model
        sometimes adds. If after cleanup the output is empty, too long, or
        contains a newline-rich explanation, give up and return None.
      - On any exception, return None — the caller will use the original query.
      - Result is cached per original query string to avoid re-translating.
    """
    if not state.llm_translate_query or not state.llm_enabled:
        return None
    if not _has_enough_cyrillic(original, state.llm_translate_min_cyrillic_chars):
        return None

    cached = _translation_cache.get(original)
    if cached:
        return cached

    try:
        client = _make_llm_client(state)
    except Exception:
        return None

    if not client.ping():
        return None

    sys_prompt = (
        "You translate Russian search queries into concise English search queries.\n\n"
        "Rules:\n"
        "1. Output ONLY the English query, on a single line.\n"
        "2. No quotes, no explanations, no labels like 'Translation:' or 'English:'.\n"
        "3. Keep names, identifiers, dates, numbers and units exactly as written.\n"
        "4. Keep already-English terms unchanged.\n"
        "5. Use natural, neutral English suitable for general document search."
    )

    try:
        raw = client.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": original.strip()},
            ]
        )
    except Exception:
        return None

    out = (raw or "").strip()
    # Common LLM tics: surrounding quotes, leading labels, extra explanation lines.
    out = out.split("\n", 1)[0].strip()
    for q in ('"', "'", "«", "»", "`"):
        out = out.strip(q)
    out = out.strip()
    for label in (
        "translation:", "english:", "the translation is:", "translated:",
        "search query:", "query:",
    ):
        if out.lower().startswith(label):
            out = out[len(label):].strip()
            for q in ('"', "'", "«", "»", "`"):
                out = out.strip(q)
            out = out.strip()
            break

    if not out or len(out) > 500 or _has_enough_cyrillic(out, 4):
        # Either empty, suspiciously long, or still mostly Russian — give up.
        return None

    _translation_cache[original] = out
    return out


def _make_llm_client(state: AppState) -> OllamaChat:
    return OllamaChat(
        OllamaChatConfig(
            model=state.llm_model,
            base_url=state.llm_base_url,
            timeout_s=int(state.llm_timeout_s),
            temperature=float(state.llm_temperature),
            top_p=float(state.llm_top_p),
            num_predict=int(state.llm_num_predict),
        )
    )


def _expand_context_items(
    state: AppState, items: Sequence[ContextItem]
) -> list[ContextItem]:
    """
    Step 3.1: widen each retrieved chunk to +/- N surrounding lines on the same page,
    and merge chunks coming from the same (rel_path, page_no).

    Important properties (designed for low risk):
      - Read-only DB access (SELECTs only, no schema/data changes).
      - Does not touch retrieval scores or printed plan/top-N — only changes what we
        hand to the LLM via _call_llm_with_context.
      - On any failure (DB error, missing lines, items with no line range) falls back
        to the original chunk text so the LLM call never breaks.
      - Order is preserved by the best-ranked item of each group.
    """
    if not items:
        return list(items)

    expand_n = max(0, int(state.llm_expand_lines))
    max_chars = max(200, int(state.llm_expand_max_chars_per_item))

    # Group items by (rel_path, page_no), remembering the best (lowest) rank.
    groups: dict[tuple[str, int], list[tuple[int, ContextItem]]] = {}
    for rank, it in enumerate(items):
        key = (str(it.citation.rel_path), int(it.citation.page_no))
        groups.setdefault(key, []).append((rank, it))

    out_with_rank: list[tuple[int, ContextItem]] = []
    conn = connect(state.db_path)
    try:
        for (rel_path, page_no), group in groups.items():
            # Sort within the group by rank; first item gives us best score/channel.
            group.sort(key=lambda rp: rp[0])
            best_rank = group[0][0]
            best_score = group[0][1].score
            best_channel = group[0][1].channel

            starts = [int(it.citation.start_line_no or 0) for _, it in group]
            ends = [int(it.citation.end_line_no or 0) for _, it in group]
            valid_starts = [s for s in starts if s > 0]
            valid_ends = [e for e in ends if e > 0]

            if not valid_starts or not valid_ends:
                # Pages-target hit (no line range) — keep as-is.
                out_with_rank.append((best_rank, group[0][1]))
                continue

            expanded_start = max(1, min(valid_starts) - expand_n)
            expanded_end = max(valid_ends) + expand_n

            try:
                rows = conn.execute(
                    """
                    SELECT l.line_no, l.text
                      FROM lines l
                      JOIN pages p ON p.page_id = l.page_id
                      JOIN documents d ON d.doc_id = p.doc_id
                     WHERE d.rel_path = ? AND p.page_no = ?
                       AND l.line_no BETWEEN ? AND ?
                     ORDER BY l.line_no
                    """,
                    (rel_path, int(page_no), int(expanded_start), int(expanded_end)),
                ).fetchall()
            except Exception:
                rows = []

            if not rows:
                # Fallback: stitch the raw chunk texts from this group.
                fallback_text = "\n".join(it.text for _, it in group if it.text)
                if len(fallback_text) > max_chars:
                    fallback_text = fallback_text[: max_chars - 1].rstrip() + "…"
                out_with_rank.append(
                    (
                        best_rank,
                        ContextItem(
                            citation=Citation(
                                rel_path=rel_path,
                                page_no=int(page_no),
                                start_line_no=min(valid_starts),
                                end_line_no=max(valid_ends),
                            ),
                            text=fallback_text,
                            channel=best_channel,
                            score=best_score,
                        ),
                    )
                )
                continue

            actual_start = int(rows[0][0])
            actual_end = int(rows[-1][0])
            text = "\n".join(str(t) for _, t in rows)
            if len(text) > max_chars:
                text = text[: max_chars - 1].rstrip() + "…"

            out_with_rank.append(
                (
                    best_rank,
                    ContextItem(
                        citation=Citation(
                            rel_path=rel_path,
                            page_no=int(page_no),
                            start_line_no=actual_start,
                            end_line_no=actual_end,
                        ),
                        text=text,
                        channel=best_channel,
                        score=best_score,
                    ),
                )
            )
    finally:
        conn.close()

    out_with_rank.sort(key=lambda x: x[0])
    return [it for _, it in out_with_rank]


def _items_to_llm_context(
    items: Sequence[ContextItem], max_chunks: int
) -> list[LLMContextItem]:
    """
    Convert our ContextItem list into the minimal shape the LLM module wants.
    Bracket labels are 1-based and match what we print to the user.
    """
    out: list[LLMContextItem] = []
    for i, it in enumerate(items[: max(1, int(max_chunks))], start=1):
        out.append(
            LLMContextItem(
                label=str(i),
                citation=it.citation.format(),
                text=str(it.text or ""),
            )
        )
    return out


def _call_llm_with_context(
    state: AppState, query: str, items: Sequence[ContextItem]
) -> Optional[str]:
    """
    Run the LLM with retrieved context. Streams to stdout if state.llm_stream is True.
    Returns the full answer text, or None on failure (error is printed).
    """
    if not items and state.llm_skip_if_no_context:
        print("[LLM skipped: no context items returned by retrieval]\n")
        return None

    try:
        client = _make_llm_client(state)
    except Exception as e:
        print(f"LLM init error: {type(e).__name__}: {e}\n")
        return None

    if not client.ping():
        print(
            f"LLM unreachable: cannot reach Ollama at {state.llm_base_url}. "
            f"Start it (`ollama serve`) or set :llm base_url, "
            f"or disable with :llm off.\n"
        )
        return None

    # Step 3.1: optionally expand each chunk to its surrounding +/- N lines on the
    # same page, merging chunks from the same (file, page). This only affects what
    # goes into the LLM; the printed retrieval results stay unchanged.
    items_for_llm: Sequence[ContextItem] = items
    expanded_used = False
    if state.llm_expand_context:
        try:
            items_for_llm = _expand_context_items(state, items)
            expanded_used = True
        except Exception as e:
            print(
                f"[context expansion failed: {type(e).__name__}: {e} "
                f"— falling back to raw retrieval chunks]"
            )
            items_for_llm = items

    llm_items = _items_to_llm_context(items_for_llm, state.llm_max_chunks)
    # When expansion is on, allow a larger per-item char cap so we don't truncate
    # the paragraph the LLM was supposed to see.
    eff_max_chars_per_item = int(
        state.llm_expand_max_chars_per_item
        if expanded_used
        else state.llm_max_chars_per_item
    )

    # Transparency: print what the LLM is actually about to receive AFTER expansion.
    # Without this, a citation like "[2] 195:1-5" can be confusing when the answer
    # references content from line 6+ (which the LLM saw via expansion but the user
    # does not see in the retrieval table above).
    if state.llm_show_expanded and expanded_used and llm_items:
        print("LLM context (after expansion)")
        print("-----------------------------")
        # Build a quick lookup from expanded citation back to the original retrieval
        # chunks that merged into it. Two items match if same (rel_path, page_no) and
        # the expanded range covers the original range.
        for i, llm_it in enumerate(llm_items, start=1):
            # Find original chunks that contributed
            originals: list[str] = []
            try:
                cit_parts = llm_it.citation.split(":")
                if len(cit_parts) >= 3:
                    expanded_path = cit_parts[0]
                    expanded_page = int(cit_parts[1])
                    rng = cit_parts[2].split("-")
                    exp_start = int(rng[0])
                    exp_end = int(rng[-1])
                else:
                    expanded_path, expanded_page, exp_start, exp_end = None, None, None, None
            except Exception:
                expanded_path, expanded_page, exp_start, exp_end = None, None, None, None

            for orig in items:
                if (
                    expanded_path is not None
                    and orig.citation.rel_path == expanded_path
                    and orig.citation.page_no == expanded_page
                    and (orig.citation.start_line_no or 0) >= (exp_start or 0)
                    and (orig.citation.end_line_no or 0) <= (exp_end or 10**9)
                ):
                    originals.append(
                        f"{orig.citation.start_line_no}-{orig.citation.end_line_no}"
                    )

            origin_note = (
                f" (expanded from lines {', '.join(originals)})" if originals else ""
            )
            text_len = len(llm_it.text or "")
            print(f"[{i}] {llm_it.citation}{origin_note}  [{text_len} chars]")
        print()

    print("Answer")
    print("------")
    sys.stdout.flush()

    writer: Optional[Callable[[str], None]] = None
    if state.llm_stream:
        def _writer(piece: str) -> None:
            sys.stdout.write(piece)
            sys.stdout.flush()
        writer = _writer

    try:
        answer = answer_with_context(
            client,
            query,
            llm_items,
            max_total_context_chars=int(state.llm_max_context_chars),
            max_chars_per_item=eff_max_chars_per_item,
            stream_writer=writer,
        )
    except LLMError as e:
        print(f"\nLLM error: {e}\n")
        return None
    except Exception as e:
        print(f"\nLLM error: {type(e).__name__}: {e}\n")
        return None

    if state.llm_stream:
        print()  # newline after streaming
    else:
        print(answer)
    print()

    # Append the (query, retrieval, answer) tuple to data/answers.jsonl so the
    # user has a ready-made appendix for the thesis. Failure here must never
    # break the answer flow.
    try:
        log_path = state.project_root / "data" / "answers.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # `query` is what _call_llm_with_context received (the original query).
        # If a translation was used for retrieval, it lives in _translation_cache.
        retrieval_query = _translation_cache.get(query, query)
        record = {
            "ts": int(time.time()),
            "query": query,
            "retrieval_query": retrieval_query,
            "translated": retrieval_query != query,
            "mode": state.mode,
            "llm_model": state.llm_model,
            "semantic_backend": state.semantic_backend,
            "semantic_model": (
                state.semantic_ollama_model
                if state.semantic_backend in ("ollama", "auto")
                else state.semantic_st_model
            ),
            "expand_context": bool(state.llm_expand_context),
            "expand_lines": int(state.llm_expand_lines) if state.llm_expand_context else 0,
            "context_count": len(llm_items),
            "citations": [it.citation for it in llm_items],
            "answer": answer,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return answer


# -----------------------------
# Printing
# -----------------------------


def _print_context_items(items: Sequence[ContextItem]) -> None:
    if not items:
        print("No results.\n")
        return

    for i, it in enumerate(items, start=1):
        score_s = f"{it.score:.6f}" if isinstance(it.score, (float, int)) else "n/a"
        print(f"{i:>2}. [{it.channel}] score={score_s}  {it.citation.format()}")
        for line in (it.text or "").split("\n"):
            if line.strip():
                print(f"    {line}")
        print()


def _format_config(state: AppState) -> str:
    sem = (
        f"{state.semantic_backend} "
        f"(hash_dim={state.semantic_hash_dim}, ollama_model={state.semantic_ollama_model}, st_model={state.semantic_st_model})"
    )
    hyb = (
        f"lex_k={state.hybrid_lex_k}, sem_k={state.hybrid_sem_k}, final={state.limit}, "
        f"rrf_k={state.hybrid_rrf_k}, w_lex={state.hybrid_w_lex}, w_sem={state.hybrid_w_sem}"
    )
    llm = (
        f"{_fmt_bool(state.llm_enabled)} (model={state.llm_model}, "
        f"url={state.llm_base_url}, temp={state.llm_temperature}, "
        f"max_chunks={state.llm_max_chunks}, stream={_fmt_bool(state.llm_stream)}, "
        f"expand={_fmt_bool(state.llm_expand_context)}/+-{state.llm_expand_lines}ln/"
        f"{state.llm_expand_max_chars_per_item}ch, "
        f"translate={_fmt_bool(state.llm_translate_query)})"
    )
    return textwrap.dedent(
        f"""
        db:            {state.db_path}
        mode:          {state.mode}
        target:        {state.target}
        limit:         {state.limit}
        context_lines: {state.context_lines}
        query_mode:    {state.query_mode}
        typos:         {_fmt_bool(state.typos)}
        synonyms:      {_fmt_bool(state.synonyms)}
        morphology:    {_fmt_bool(state.morphology)}
        semantic:      {sem}
        hybrid:        {hyb}
        llm:           {llm}
        """
    ).strip()


def _help_text() -> str:
    return textwrap.dedent(
        """
        Commands
        --------
        :help                         show this help
        :config                       show current settings
        :quit / :exit                 exit the app

        Index-time
        ----------
        :index [ocr]                  run Step A + Step B (optionally with OCR)
        :embed [backend] [args...]    run Step D (chunks + embeddings) as a script
                                     examples:
                                       :embed hash 128
                                       :embed ollama nomic-embed-text

        Query-time mode
        --------------
        :mode lexical|semantic|hybrid set retrieval mode (default: lexical)
        :target lines|pages           lexical target (default: lines)
        :limit N                      number of results to print
        :context N                    context lines around a lexical line match (lines target)

        Query features
        --------------
        :typos on|off                 conservative typo correction for ordinary RU/EN words
        :syn on|off                   synonyms expansion
        :morph on|off                 morphology expansion
        :qmode auto|simple|advanced   query mode (advanced passes FTS5 syntax)

        Semantic settings (used by :mode semantic / hybrid)
        ---------------------------------------------------
        :sem backend auto|ollama|sentence_transformers|hash
        :sem hash_dim N
        :sem ollama_model NAME
        :sem rel_path PATH            restrict semantic retrieval to a single document
        :sem status                   show current model and embeddings present in DB

        LLM answer (step 7) - automatic after every plain query
        -------------------------------------------------------
        :llm on|off                   enable/disable LLM answer
        :llm model NAME               set chat model (e.g. t-lite-it, t-pro-it)
        :llm base_url URL             Ollama base URL (default http://localhost:11434)
        :llm temperature F            sampling temperature (default 0.2)
        :llm max_chunks N             how many retrieval items to feed the LLM
        :llm num_predict N            max output tokens
        :llm timeout SECONDS          per-request timeout (raise for 32B models)
        :llm stream on|off            stream tokens to stdout (default on)
        :llm expand on|off            widen each retrieved chunk to surrounding lines
                                       on the same page before sending to LLM (default on)
        :llm expand_lines N           +/- N lines around each chunk (default 15)
        :llm expand_chars N           per-item char cap when expansion is on (default 4000)
        :llm translate on|off         auto-translate Russian queries to English via the
                                       chat LLM before retrieval (default on)
        :llm show_expanded on|off     print what the LLM received after expansion
                                       (transparency for "where did this number come from")
                                       (default on)
        :llm status                   ping Ollama and list installed models

        Translation override
        --------------------
        :translate                    show current translation cache
        :translate clear              wipe the translation cache
        :translate <english>          override translation for the LAST query and re-issue it

        Run a query
        -----------
        Just type a query line (not starting with ':') and press Enter.
        Retrieval runs first, then (if :llm is on) the LLM answers with citations.
        """
    ).strip()


# -----------------------------
# Command handling
# -----------------------------


class CommandError(RuntimeError):
    pass


def _parse_on_off(val: str) -> bool:
    v = (val or "").strip().lower()
    if v in ("on", "1", "true", "yes", "y"):
        return True
    if v in ("off", "0", "false", "no", "n"):
        return False
    raise CommandError("expected on|off")


def _handle_index(state: AppState, args: list[str]) -> None:
    """
    :index [ocr]
    Runs Step A + Step B as subprocesses.
    """
    use_ocr = False
    if args and args[0].lower() == "ocr":
        use_ocr = True

    _ensure_db_initialized(state)

    print("Running Step A (scan corpus)...")
    rc = _run_script(
        state.project_root, ["scripts/step_a_scan_corpus.py", "--prune-missing"]
    )
    if rc != 0:
        raise CommandError(f"Step A failed (exit code {rc})")

    print("Running Step B (index pages/lines)...")
    step_b = ["scripts/step_b_index_pages.py"]
    if use_ocr:
        step_b += ["--ocr"]
    rc = _run_script(state.project_root, step_b)
    if rc != 0:
        raise CommandError(f"Step B failed (exit code {rc})")

    print("Index-time done.\n")


def _handle_embed(state: AppState, args: list[str]) -> None:
    """
    :embed [backend] [args...]
    Runs Step D (embed chunks) as subprocess, with convenient shorthands.
    """
    _ensure_db_initialized(state)

    backend = args[0].lower() if args else state.semantic_backend
    if backend not in ("auto", "ollama", "sentence_transformers", "hash"):
        raise CommandError("usage: :embed [auto|ollama|sentence_transformers|hash] ...")

    step_d = ["scripts/step_d_embed_chunks.py"]

    # Default embedder settings from state; allow overrides via args
    if backend == "hash":
        dim = state.semantic_hash_dim
        if len(args) >= 2:
            dim = _safe_int(args[1], "hash_dim")
        step_d += ["--embedder", "hash", "--hash-dim", str(dim)]
    elif backend == "ollama":
        model = state.semantic_ollama_model
        if len(args) >= 2:
            model = str(args[1])
        step_d += ["--embedder", "ollama", "--ollama-model", model]
    elif backend == "sentence_transformers":
        model = state.semantic_st_model
        if len(args) >= 2:
            model = str(args[1])
        step_d += ["--embedder", "sentence_transformers", "--st-model", model]
    else:
        step_d += ["--embedder", "auto"]

    # Keep default behavior safe: incremental + reuse chunks. Users can pass extra flags.
    # If the user provides extra args, append them verbatim after the shorthand.
    if len(args) > 2:
        step_d += args[2:]

    print("Running Step D (build chunks + embeddings)...")
    rc = _run_script(state.project_root, step_d)
    if rc != 0:
        raise CommandError(f"Step D failed (exit code {rc})")
    print("Embeddings build done.\n")


def _handle_command(state: AppState, line: str) -> bool:
    """
    Handle a command line starting with ':'.
    Returns True to continue, False to exit.
    """
    raw = line.strip()
    if not raw.startswith(":"):
        return True

    parts = shlex.split(raw[1:].strip())
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("quit", "exit"):
        return False

    if cmd == "help":
        _print_block("Help", _help_text())
        return True

    if cmd == "config":
        _print_block("Config", _format_config(state))
        return True

    if cmd == "mode":
        if not args:
            raise CommandError("usage: :mode lexical|semantic|hybrid")
        m = args[0].lower()
        if m not in ("lexical", "semantic", "hybrid"):
            raise CommandError("usage: :mode lexical|semantic|hybrid")
        state.mode = m  # type: ignore[assignment]
        print(f"mode = {state.mode}\n")
        return True

    if cmd == "target":
        if not args:
            raise CommandError("usage: :target lines|pages")
        t = args[0].lower()
        if t not in ("lines", "pages"):
            raise CommandError("usage: :target lines|pages")
        state.target = t  # type: ignore[assignment]
        print(f"target = {state.target}\n")
        return True

    if cmd == "limit":
        if not args:
            raise CommandError("usage: :limit N")
        state.limit = _safe_int(args[0], "limit")
        print(f"limit = {state.limit}\n")
        return True

    if cmd == "context":
        if not args:
            raise CommandError("usage: :context N")
        state.context_lines = _safe_int(args[0], "context")
        print(f"context_lines = {state.context_lines}\n")
        return True

    if cmd == "typos":
        if not args:
            raise CommandError("usage: :typos on|off")
        state.typos = _parse_on_off(args[0])
        print(f"typos = {_fmt_bool(state.typos)}\n")
        return True

    if cmd in ("syn", "synonyms"):
        if not args:
            raise CommandError("usage: :syn on|off")
        state.synonyms = _parse_on_off(args[0])
        print(f"synonyms = {_fmt_bool(state.synonyms)}\n")
        return True

    if cmd in ("morph", "morphology"):
        if not args:
            raise CommandError("usage: :morph on|off")
        state.morphology = _parse_on_off(args[0])
        print(f"morphology = {_fmt_bool(state.morphology)}\n")
        return True

    if cmd in ("qmode", "query_mode"):
        if not args:
            raise CommandError("usage: :qmode auto|simple|advanced")
        q = args[0].lower()
        if q not in ("auto", "simple", "advanced"):
            raise CommandError("usage: :qmode auto|simple|advanced")
        state.query_mode = q  # type: ignore[assignment]
        print(f"query_mode = {state.query_mode}\n")
        return True

    if cmd == "sem":
        if not args:
            raise CommandError(
                "usage: :sem backend <auto|ollama|sentence_transformers|hash> | "
                ":sem hash_dim N | :sem ollama_model NAME | :sem rel_path PATH"
            )
        sub = args[0].lower()
        if sub in ("backend",):
            if len(args) < 2:
                raise CommandError(
                    "usage: :sem backend auto|ollama|sentence_transformers|hash"
                )
            b = args[1].lower()
            if b not in ("auto", "ollama", "sentence_transformers", "hash"):
                raise CommandError(
                    "usage: :sem backend auto|ollama|sentence_transformers|hash"
                )
            state.semantic_backend = b  # type: ignore[assignment]
            print(f"semantic_backend = {state.semantic_backend}\n")
            return True
        if sub == "hash_dim":
            if len(args) < 2:
                raise CommandError("usage: :sem hash_dim N")
            state.semantic_hash_dim = _safe_int(args[1], "hash_dim")
            print(f"semantic_hash_dim = {state.semantic_hash_dim}\n")
            return True
        if sub == "ollama_model":
            if len(args) < 2:
                raise CommandError("usage: :sem ollama_model NAME")
            state.semantic_ollama_model = str(args[1])
            print(f"semantic_ollama_model = {state.semantic_ollama_model}\n")
            return True
        if sub == "rel_path":
            if len(args) < 2:
                state.semantic_rel_path = None
                print("semantic_rel_path = (none)\n")
            else:
                state.semantic_rel_path = str(args[1])
                print(f"semantic_rel_path = {state.semantic_rel_path}\n")
            return True
        if sub == "status":
            # Show: current config + breakdown of embeddings in the DB by model.
            # Useful for spotting "I switched to bge-m3 but never ran :embed".
            conn = connect(state.db_path)
            try:
                rows = conn.execute(
                    "SELECT model, COUNT(*) FROM chunk_embeddings "
                    "GROUP BY model ORDER BY COUNT(*) DESC"
                ).fetchall()
                total_chunks = conn.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
            finally:
                conn.close()

            cur_model = (
                state.semantic_ollama_model
                if state.semantic_backend in ("ollama", "auto")
                else state.semantic_st_model
                if state.semantic_backend == "sentence_transformers"
                else f"hash:{state.semantic_hash_dim}"
            )
            print(f"backend: {state.semantic_backend}")
            print(f"model:   {cur_model}")
            print(f"chunks total in DB: {total_chunks}")
            print("embeddings by model label:")
            if rows:
                for model, n in rows:
                    pct = (n / total_chunks * 100.0) if total_chunks else 0.0
                    print(f"  {n:>6}  ({pct:5.1f}%)  {model}")
            else:
                print("  (no embeddings yet — run :embed)")
            print()
            return True
        raise CommandError("unknown :sem subcommand")

    if cmd == "llm":
        if not args:
            raise CommandError(
                "usage: :llm on|off | :llm model NAME | :llm base_url URL | "
                ":llm temperature F | :llm max_chunks N | :llm stream on|off | "
                ":llm status"
            )
        sub = args[0].lower()
        if sub in ("on", "off"):
            state.llm_enabled = sub == "on"
            print(f"llm = {_fmt_bool(state.llm_enabled)}\n")
            return True
        if sub == "model":
            if len(args) < 2:
                raise CommandError("usage: :llm model NAME")
            state.llm_model = str(args[1])
            print(f"llm_model = {state.llm_model}\n")
            return True
        if sub in ("base_url", "url"):
            if len(args) < 2:
                raise CommandError("usage: :llm base_url URL")
            state.llm_base_url = str(args[1])
            print(f"llm_base_url = {state.llm_base_url}\n")
            return True
        if sub in ("temperature", "temp"):
            if len(args) < 2:
                raise CommandError("usage: :llm temperature F")
            try:
                state.llm_temperature = float(args[1])
            except Exception as e:
                raise CommandError(f"bad temperature: {args[1]!r}") from e
            print(f"llm_temperature = {state.llm_temperature}\n")
            return True
        if sub == "max_chunks":
            if len(args) < 2:
                raise CommandError("usage: :llm max_chunks N")
            state.llm_max_chunks = _safe_int(args[1], "max_chunks")
            print(f"llm_max_chunks = {state.llm_max_chunks}\n")
            return True
        if sub == "num_predict":
            if len(args) < 2:
                raise CommandError("usage: :llm num_predict N")
            state.llm_num_predict = _safe_int(args[1], "num_predict")
            print(f"llm_num_predict = {state.llm_num_predict}\n")
            return True
        if sub == "timeout":
            if len(args) < 2:
                raise CommandError("usage: :llm timeout SECONDS")
            state.llm_timeout_s = _safe_int(args[1], "timeout")
            print(f"llm_timeout_s = {state.llm_timeout_s}\n")
            return True
        if sub == "stream":
            if len(args) < 2:
                raise CommandError("usage: :llm stream on|off")
            state.llm_stream = _parse_on_off(args[1])
            print(f"llm_stream = {_fmt_bool(state.llm_stream)}\n")
            return True
        if sub == "expand":
            if len(args) < 2:
                raise CommandError("usage: :llm expand on|off")
            state.llm_expand_context = _parse_on_off(args[1])
            print(f"llm_expand_context = {_fmt_bool(state.llm_expand_context)}\n")
            return True
        if sub == "translate":
            if len(args) < 2:
                raise CommandError("usage: :llm translate on|off")
            state.llm_translate_query = _parse_on_off(args[1])
            print(f"llm_translate_query = {_fmt_bool(state.llm_translate_query)}\n")
            return True
        if sub == "show_expanded":
            if len(args) < 2:
                raise CommandError("usage: :llm show_expanded on|off")
            state.llm_show_expanded = _parse_on_off(args[1])
            print(f"llm_show_expanded = {_fmt_bool(state.llm_show_expanded)}\n")
            return True
        if sub == "expand_lines":
            if len(args) < 2:
                raise CommandError("usage: :llm expand_lines N")
            state.llm_expand_lines = _safe_int(args[1], "expand_lines")
            print(f"llm_expand_lines = {state.llm_expand_lines}\n")
            return True
        if sub == "expand_chars":
            if len(args) < 2:
                raise CommandError("usage: :llm expand_chars N")
            state.llm_expand_max_chars_per_item = _safe_int(args[1], "expand_chars")
            print(
                f"llm_expand_max_chars_per_item = {state.llm_expand_max_chars_per_item}\n"
            )
            return True
        if sub == "status":
            client = _make_llm_client(state)
            up = client.ping()
            print(f"Ollama at {state.llm_base_url}: {'reachable' if up else 'UNREACHABLE'}")
            if up:
                models = client.list_models()
                if models:
                    print("Installed models:")
                    for m in models:
                        marker = " <- current" if m == state.llm_model else ""
                        print(f"  - {m}{marker}")
                else:
                    print("No models installed (run e.g. `ollama pull t-lite-it`).")
            print(f"current model: {state.llm_model}\n")
            return True
        raise CommandError(f"unknown :llm subcommand: {sub}")

    if cmd == "translate":
        # Top-level command for manual translation control.
        #   :translate              show current cache
        #   :translate clear        wipe the cache
        #   :translate <english>    override the translation for the LAST query
        if not args:
            if not _translation_cache:
                print("translation cache: (empty)\n")
            else:
                print("translation cache:")
                for k, v in _translation_cache.items():
                    print(f"  {k!r}\n    -> {v!r}")
                print()
            return True
        if args[0].lower() == "clear":
            n = len(_translation_cache)
            _translation_cache.clear()
            print(f"translation cache cleared ({n} entries)\n")
            return True
        if state.last_query is None:
            raise CommandError(
                "no last query to override — first issue a query, then :translate <english>"
            )
        override = " ".join(args).strip()
        if not override:
            raise CommandError("empty translation override")
        _translation_cache[state.last_query] = override
        print(
            f"override saved for last query:\n"
            f"  {state.last_query!r}\n"
            f"  -> {override!r}\n"
            f"(re-issue the same query to use it)\n"
        )
        return True

    if cmd == "index":
        _handle_index(state, args)
        return True

    if cmd == "embed":
        _handle_embed(state, args)
        return True

    raise CommandError(f"unknown command: :{cmd}")


# -----------------------------
# REPL loop
# -----------------------------


def _run_query(state: AppState, query: str) -> None:
    state.last_query = query
    state.last_context = []

    # Ensure DB exists (read-time). We won't force indexing here.
    if not state.db_path.exists():
        print(f"DB not found: {state.db_path}")
        print("Run :index first to build the index.\n")
        return

    _ensure_db_initialized(state)

    # Step 3.3: translate Russian queries to English BEFORE retrieval.
    # The original Russian is still passed to the LLM for the final answer
    # so the user sees a Russian reply.
    original_query = query
    translated_query = _maybe_translate_query(state, query)
    effective_query = translated_query if translated_query else query

    if translated_query and state.show_plan:
        print("Query translation")
        print("-----------------")
        print(f"original:   {original_query!r}")
        print(f"translated: {translated_query!r}")
        print(
            "(retrieval uses the English form; the LLM answers using the original)"
        )
        print()

    if state.mode == "lexical":
        try:
            plan, items = _lexical_context(state, effective_query)
        except SearchError as e:
            print(f"Search error: {e}\n")
            return
        if state.show_plan:
            # `plan` is a dataclass; print its essential bits
            print("Query plan")
            print("---------")
            print(f"raw:        {getattr(plan, 'raw', '')!r}")
            print(f"normalized: {getattr(plan, 'normalized', '')!r}")
            print(
                f"mode:       {getattr(plan, 'mode', '')} (advanced={bool(getattr(plan, 'is_advanced', False))})"
            )
            print(f"fts:        {getattr(plan, 'fts', '')!r}")
            print()
        _print_context_items(items)
        state.last_context = items
        if state.llm_enabled:
            _call_llm_with_context(state, original_query, items)
        return

    if state.mode == "hybrid":
        try:
            hyb_plan, items = _hybrid_context(state, effective_query)
        except Exception as e:
            print(f"Hybrid error: {type(e).__name__}: {e}\n")
            return

        if state.show_plan:
            print("Hybrid plan")
            print("-----------")
            # lexical plan is present
            lp = getattr(hyb_plan, "lexical_plan", None)
            if lp is not None:
                print(
                    f"lexical:   mode={getattr(lp, 'mode', '')} fts={getattr(lp, 'fts', '')!r}"
                )
            print(
                f"semantic:  enabled={bool(getattr(hyb_plan, 'semantic_enabled', False))} model_label={getattr(hyb_plan, 'semantic_model_label', None)!r}"
            )
            print(
                f"fusion:    {getattr(hyb_plan, 'fusion', 'rrf')} rrf_k={getattr(hyb_plan, 'rrf_k', '')} w_lex={getattr(hyb_plan, 'w_lex', '')} w_sem={getattr(hyb_plan, 'w_sem', '')}"
            )
            print()
        _print_context_items(items)
        state.last_context = items
        if state.llm_enabled:
            _call_llm_with_context(state, original_query, items)
        return

    # semantic-only mode: minimal behavior (requires Step D)
    if state.mode == "semantic":
        try:
            embedder = _create_semantic_embedder(state)
            model_label = stable_model_label_for_embedder(embedder)
        except EmbeddingsError as e:
            print(f"Semantic error: failed to create embedder: {e}\n")
            return

        conn = connect(state.db_path)
        try:
            from src.index.hybrid import semantic_retrieve_sqlite  # lazy import

            model_label, cands = semantic_retrieve_sqlite(
                conn,
                embedder=embedder,
                query=effective_query,
                top_k=int(state.limit),
                min_score=float(state.semantic_min_score),
                rel_path=state.semantic_rel_path,
            )
        except Exception as e:
            print(f"Semantic error: {type(e).__name__}: {e}\n")
            return
        finally:
            conn.close()

        if state.show_plan:
            print("Semantic plan")
            print("-------------")
            print(
                f"embedder:   {embedder.info().backend}  model_label={model_label!r}  dim={embedder.info().dim}"
            )
            if state.semantic_rel_path:
                print(f"rel_path:   {state.semantic_rel_path!r}")
            print()

        items: list[ContextItem] = []
        for c in cands:
            items.append(
                ContextItem(
                    citation=Citation(
                        rel_path=str(c.key.rel_path),
                        page_no=int(c.key.page_no),
                        start_line_no=int(c.key.start_line_no),
                        end_line_no=int(c.key.end_line_no),
                    ),
                    text=str(c.text or ""),
                    channel="semantic",
                    score=float(c.score),
                )
            )

        _print_context_items(items)
        state.last_context = items
        if state.llm_enabled:
            _call_llm_with_context(state, original_query, items)
        return

    print(f"Unknown mode: {state.mode}\n")


def repl(state: AppState) -> int:
    # Startup readiness checks / interactive prompts (safe defaults).
    # This keeps the UX "app-like" without breaking existing scripts.
    _startup_readiness_check(state)

    _print_block(
        "older REPL",
        textwrap.dedent(
            f"""
            Project root: {state.project_root}
            DB:           {state.db_path}

            Type :help for commands.
            Default mode: lexical (FTS5/BM25) - requires Step B.
            """
        ).strip(),
    )

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not line:
            continue

        if line.startswith(":"):
            try:
                keep = _handle_command(state, line)
            except CommandError as e:
                print(f"Command error: {e}\n")
                continue
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}\n")
                continue

            if not keep:
                print("bye")
                return 0
            continue

        _run_query(state, line)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive REPL for older search project."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite DB (default: <project_root>/data/index.sqlite)",
    )
    parser.add_argument(
        "--mode",
        choices=["lexical", "semantic", "hybrid"],
        default="lexical",
        help="Default mode on startup.",
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Do not print query plans.",
    )
    parser.add_argument(
        "--no-startup-prompts",
        action="store_true",
        help="Disable startup DB readiness checks and prompts to run :index / :embed.",
    )
    parser.add_argument(
        "--llm",
        choices=["on", "off"],
        default=None,
        help="Enable/disable LLM answer step (default: on).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Ollama chat model name (e.g. t-lite-it, t-pro-it).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--sem-backend",
        choices=["auto", "ollama", "sentence_transformers", "hash"],
        default=None,
        help="Semantic backend for hybrid/semantic mode.",
    )
    parser.add_argument(
        "--sem-model",
        default=None,
        help=(
            "Semantic model name (e.g. nomic-embed-text, bge-m3 for ollama; "
            "all-MiniLM-L6-v2 for sentence_transformers). For 'hash' this is "
            "interpreted as the dimension (integer)."
        ),
    )

    args = parser.parse_args(argv)

    project_root = _project_root_from_this_file()
    db_path = Path(args.db) if args.db else _default_db_path(project_root)

    state = AppState(project_root=project_root, db_path=db_path)
    state.mode = args.mode  # type: ignore[assignment]
    state.show_plan = not bool(args.no_plan)
    state.startup_prompts = not bool(args.no_startup_prompts)
    if args.llm is not None:
        state.llm_enabled = args.llm == "on"
    if args.llm_model:
        state.llm_model = str(args.llm_model)
    if args.llm_base_url:
        state.llm_base_url = str(args.llm_base_url)
    if args.sem_backend:
        state.semantic_backend = args.sem_backend  # type: ignore[assignment]
    if args.sem_model:
        if state.semantic_backend in ("ollama", "auto"):
            state.semantic_ollama_model = str(args.sem_model)
        elif state.semantic_backend == "sentence_transformers":
            state.semantic_st_model = str(args.sem_model)
        elif state.semantic_backend == "hash":
            try:
                state.semantic_hash_dim = int(args.sem_model)
            except ValueError:
                print(
                    f"warning: --sem-model {args.sem_model!r} is not an integer; "
                    f"ignored for hash backend (kept dim={state.semantic_hash_dim})"
                )

    return repl(state)


if __name__ == "__main__":
    raise SystemExit(main())
