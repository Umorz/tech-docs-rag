from __future__ import annotations

"""
Query-time search module (MVP): FTS5 MATCH + BM25 ranking over indexed PDF text.

This module is intentionally small and dependency-free (stdlib only). It is meant to
sit *on top* of the existing Index-time pipeline:

- Step A: documents registry in SQLite (table: documents)
- Step B: extracted text stored as pages/lines + FTS5 indices (pages_fts, lines_fts)

What this module provides today (per Указания.md):
  (1) Формирование запроса (minimal, safe normalization + simple boolean assembly)
  (4) Булев поиск / BM25 (SQLite FTS5 MATCH + bm25())

Designed to be extended later with:
  (2) typo correction
  (3) synonyms / morphology
  (5) embeddings (semantic retrieval)
  (6) reranking
"""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from . import text_clean

SearchTarget = Literal["lines", "pages"]
QueryMode = Literal["auto", "simple", "advanced"]


# -----------------------------
# Public result models
# -----------------------------


@dataclass(frozen=True)
class QueryPlan:
    """
    A small, explicit artifact that captures how a user query turns into an FTS5 query.

    Notes:
      - `raw` is always the user input.
      - `normalized` / `fts` reflect the FINAL query actually used for retrieval.
        (They may include typo correction and/or synonym/morphology expansion when enabled.)
      - `*_before_*` and `*_corrections/*_expansions` preserve explainability.
    """

    raw: str
    normalized: str
    fts: str
    mode: QueryMode
    is_advanced: bool

    # Typo correction (step 2) explainability
    normalized_before_typos: Optional[str] = None
    fts_before_typos: Optional[str] = None
    # (original, corrected, score_0_100, method, edits)
    typo_corrections: tuple[tuple[str, str, float, str, Optional[int]], ...] = ()

    # Synonyms / morphology expansion (step 3) explainability
    normalized_before_expansion: Optional[str] = None
    fts_before_expansion: Optional[str] = None
    # (token_or_phrase, added_terms, source_label)
    synonym_expansions: tuple[tuple[str, tuple[str, ...], str], ...] = ()
    # (token, added_variants, method_label)
    morphology_expansions: tuple[tuple[str, tuple[str, ...], str], ...] = ()


@dataclass(frozen=True)
class LineHit:
    rel_path: str
    page_no: int
    line_no: int
    text: str
    snippet: str
    score: float
    is_ocr: bool


@dataclass(frozen=True)
class PageHit:
    rel_path: str
    page_no: int
    text_len: int
    snippet: str
    score: float
    is_ocr: bool


# -----------------------------
# Query formation (MVP)
# -----------------------------


_BOOLEAN_KEYWORDS_RE = re.compile(r"(?i)\b(AND|OR|NOT|NEAR)\b")

_HAS_ADVANCED_CHARS_RE = re.compile(r"""["'():*]""")


def normalize_user_query(q: str) -> str:
    """
    Minimal, safe normalization aligned with index-time text normalization.

    Notes:
      - We do NOT force lower-case (indexing keeps case; FTS5 is typically casefolding).
      - We DO normalize unicode and stabilize micro-symbols µ/μ -> u, spaces, etc.
    """
    q = (q or "").strip()
    q = text_clean.normalize_unicode(q)
    q = q.replace("\n", " ").replace("\r", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _looks_like_advanced_fts(q: str) -> bool:
    """
    Heuristic: treat query as "advanced" if it appears to use FTS syntax.

    Advanced means: we mostly pass it through (after unicode normalization), because
    the user intentionally uses boolean operators, quotes, parentheses, wildcards, etc.
    """
    if not q:
        return False
    if _BOOLEAN_KEYWORDS_RE.search(q):
        return True
    if _HAS_ADVANCED_CHARS_RE.search(q):
        return True
    return False


def _escape_fts_phrase(s: str) -> str:
    """
    Escape a token for inclusion inside double quotes in an FTS5 query.

    In SQLite FTS queries, a double-quote inside a phrase is escaped by doubling it.
    """
    return s.replace('"', '""')


def _tokenize_simple(q: str) -> list[str]:
    """
    Tokenize a simple query into whitespace-separated terms.

    We intentionally keep punctuation within terms, but later quote terms that contain
    non-word characters to avoid FTS syntax surprises (e.g., minus, colon, etc.).
    """
    q = re.sub(r"\s+", " ", (q or "").strip())
    if not q:
        return []
    return [t for t in q.split(" ") if t]


def build_fts5_query(user_query: str, *, mode: QueryMode = "auto") -> QueryPlan:
    """
    Build an FTS5 query string from a user query.

    Modes:
      - auto: advanced if query "looks like" it uses FTS syntax; otherwise simple AND query
      - simple: always build AND query from tokens
      - advanced: pass-through (after normalization)

    Simple mode strategy (MVP):
      - split by whitespace
      - join with AND (precision-first)
      - quote terms that contain non [A-Za-z0-9_] characters
        (prevents characters like '-' from being interpreted as syntax in some contexts)
    """
    raw = user_query or ""
    normalized = normalize_user_query(raw)

    if mode not in ("auto", "simple", "advanced"):
        raise ValueError(f"Unsupported query mode: {mode}")

    if mode == "auto":
        is_adv = _looks_like_advanced_fts(normalized)
        mode_used: QueryMode = "advanced" if is_adv else "simple"
    else:
        mode_used = mode

    if mode_used == "advanced":
        # Pass-through (normalized only). If the user enters an invalid FTS query,
        # SQLite will raise an OperationalError; callers should surface it nicely.
        fts = normalized
        return QueryPlan(
            raw=raw,
            normalized=normalized,
            fts=fts,
            mode=mode_used,
            is_advanced=True,
        )

    # Simple mode
    terms = _tokenize_simple(normalized)
    if not terms:
        # An empty FTS query is not valid; return a plan that callers can reject.
        return QueryPlan(
            raw=raw,
            normalized=normalized,
            fts="",
            mode=mode_used,
            is_advanced=False,
        )

    safe_terms: list[str] = []
    for t in terms:
        # If a term contains only "word" chars, keep as is.
        # Otherwise quote it as a phrase of length 1 to avoid syntax interpretation.
        if re.fullmatch(r"\w+", t, flags=re.UNICODE):
            safe_terms.append(t)
        else:
            safe_terms.append(f'"{_escape_fts_phrase(t)}"')

    fts = " AND ".join(safe_terms)
    return QueryPlan(
        raw=raw,
        normalized=normalized,
        fts=fts,
        mode=mode_used,
        is_advanced=False,
    )


# -----------------------------
# Core FTS5 search (BM25)
# -----------------------------


@dataclass(frozen=True)
class SearchOptions:
    target: SearchTarget = "lines"
    limit: int = 20
    offset: int = 0

    # snippet formatting
    snippet_start: str = "["
    snippet_end: str = "]"
    snippet_ellipsis: str = "…"
    snippet_tokens: int = 18

    # query behavior
    query_mode: QueryMode = "auto"

    # typo correction (step 2)
    # Default: disabled (enable explicitly via CLI / calling code).
    # When enabled, the safe default behavior is "fallback":
    # - run the original query first
    # - only if it returns 0 hits, try correcting ordinary RU/EN words and re-run
    typos_enabled: bool = False
    typos_mode: str = "fallback"  # "fallback" | "always" | "off"
    typos_min_token_len: int = 4
    typos_min_score: float = 85.0
    typos_max_edits: int = 2
    typos_min_freq: int = 2
    typos_lexicon_limit: Optional[int] = None
    typos_force_rebuild: bool = False

    # synonyms / morphology (step 3)
    # Default: disabled (enable explicitly via CLI / calling code).
    # Applied only in simple-mode queries (non-advanced FTS).
    synonyms_enabled: bool = False
    synonyms_mode: str = "always"  # "always" | "fallback" | "off"
    synonyms_path: Optional[str] = None  # default: <project_root>/data/synonyms.json
    synonyms_min_token_len: int = 4
    synonyms_max_per_term: int = 6
    synonyms_include_phrases: bool = True

    morphology_enabled: bool = False
    morphology_mode: str = "always"  # "always" | "fallback" | "off"
    morphology_min_token_len: int = 4
    morphology_ru_enabled: bool = True
    morphology_en_enabled: bool = True
    morphology_ru_max_variants: int = 10
    morphology_en_max_variants: int = 6


class SearchError(RuntimeError):
    """Raised for user-facing search errors (invalid query, missing tables, etc.)."""


def _require_positive_int(name: str, value: int) -> int:
    try:
        v = int(value)
    except Exception as e:
        raise ValueError(f"{name} must be an integer, got: {value!r}") from e
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got: {v}")
    return v


def _ensure_schema_present(conn: sqlite3.Connection) -> None:
    """
    Light validation that the required tables exist.

    This prevents confusing "no such table" errors later, and makes it explicit
    that Query-time depends on having run Index-time first.
    """
    required = {
        "documents",
        "pages",
        "lines",
        "pages_fts",
        "lines_fts",
    }
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    present = {str(r[0]) for r in rows}
    missing = sorted(required - present)
    if missing:
        raise SearchError(
            "Search DB schema is missing required tables. "
            "Run index-time steps first (Step A + Step B). "
            f"Missing: {', '.join(missing)}"
        )


def _query_plan_with_typos(
    plan: QueryPlan,
    *,
    corrected_normalized: str,
    corrected_fts: str,
    corrections: Sequence[object],
) -> QueryPlan:
    """
    Return a new QueryPlan reflecting typo correction.

    We keep `raw` unchanged and set:
      - normalized/fts to the corrected versions (final used)
      - *_before_typos to the previous values
      - typo_corrections as a simple tuple-of-tuples for stable logging/printing
    """
    packed: list[tuple[str, str, float, str, Optional[int]]] = []

    for c in corrections:
        # `src.index.typos.Correction` has: original, corrected, score, method, edits
        try:
            orig = str(getattr(c, "original"))
            corr = str(getattr(c, "corrected"))
            score = float(getattr(c, "score"))
            method = str(getattr(c, "method"))
            edits = getattr(c, "edits", None)
            edits_i = int(edits) if edits is not None else None
        except Exception:
            # Fallback: keep an opaque representation
            orig = str(c)
            corr = str(c)
            score = 0.0
            method = "unknown"
            edits_i = None

        packed.append((orig, corr, score, method, edits_i))

    return QueryPlan(
        raw=plan.raw,
        normalized=corrected_normalized,
        fts=corrected_fts,
        mode=plan.mode,
        is_advanced=False,
        normalized_before_typos=plan.normalized_before_typos or plan.normalized,
        fts_before_typos=plan.fts_before_typos or plan.fts,
        typo_corrections=tuple(packed),
        normalized_before_expansion=plan.normalized_before_expansion,
        fts_before_expansion=plan.fts_before_expansion,
        synonym_expansions=plan.synonym_expansions,
        morphology_expansions=plan.morphology_expansions,
    )


def _default_project_root() -> Path:
    # search.py -> src/index/search.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def _default_synonyms_path() -> Path:
    return _default_project_root() / "data" / "synonyms.json"


# Simple in-process cache to avoid re-reading synonyms file on every query
_SYNONYMS_CACHE: Dict[str, Dict[str, Set[str]]] = {}


def _load_synonyms_map(path: Optional[str]) -> tuple[Dict[str, Set[str]], str]:
    """
    Load synonyms map from JSON file if available.

    Returns:
      (syn_map, source_label)
    """
    try:
        from . import synonyms as _syn
    except Exception:
        return {}, "none"

    p = Path(path) if path else _default_synonyms_path()
    key = str(p.resolve())

    if key in _SYNONYMS_CACHE:
        return _SYNONYMS_CACHE[key], f"file:{p.as_posix()}"

    if not p.exists():
        _SYNONYMS_CACHE[key] = {}
        return {}, f"missing:{p.as_posix()}"

    try:
        mp = _syn.load_synonyms_json(p)
    except Exception:
        mp = {}

    _SYNONYMS_CACHE[key] = mp
    return mp, f"file:{p.as_posix()}"


def _query_plan_with_expansions(
    plan: QueryPlan,
    *,
    expanded_fts: str,
    synonym_expansions: Sequence[tuple[str, tuple[str, ...], str]],
    morphology_expansions: Sequence[tuple[str, tuple[str, ...], str]],
) -> QueryPlan:
    """
    Return a new QueryPlan reflecting synonyms/morphology expansion.

    We keep:
      - raw unchanged
      - normalized unchanged (already post-typos if typos ran)
    And set:
      - fts to expanded_fts (final used)
      - *_before_expansion and expansions lists for explainability
    """
    return QueryPlan(
        raw=plan.raw,
        normalized=plan.normalized,
        fts=str(expanded_fts),
        mode=plan.mode,
        is_advanced=plan.is_advanced,
        normalized_before_typos=plan.normalized_before_typos,
        fts_before_typos=plan.fts_before_typos,
        typo_corrections=plan.typo_corrections,
        normalized_before_expansion=plan.normalized_before_expansion or plan.normalized,
        fts_before_expansion=plan.fts_before_expansion or plan.fts,
        synonym_expansions=tuple(synonym_expansions),
        morphology_expansions=tuple(morphology_expansions),
    )


def _expand_simple_query_fts(
    normalized_query: str,
    *,
    conn: sqlite3.Connection,
    opts: SearchOptions,
    apply_synonyms: bool,
    apply_morphology: bool,
) -> tuple[
    str, list[tuple[str, tuple[str, ...], str]], list[tuple[str, tuple[str, ...], str]]
]:
    """
    Expand a SIMPLE normalized query with:
      - synonyms (from data/synonyms.json)
      - morphology variants (RU via pymorphy3 if installed; EN via heuristics)

    Improvements (clean pipeline behavior):
      - Mode handling is performed by the caller; this function receives explicit
        booleans (apply_synonyms/apply_morphology).
      - Morphology variants are filtered through the corpus lexicon when available,
        to avoid widening queries with forms that never appear in the indexed corpus.
      - Phrase-aware synonyms: tries 3-gram then 2-gram phrase groups (optional).
    """
    try:
        from . import synonyms as _syn
    except Exception:
        _syn = None  # type: ignore[assignment]

    try:
        from . import morphology as _morph
    except Exception:
        _morph = None  # type: ignore[assignment]

    # Tokenize in "simple mode" (whitespace)
    tokens = (normalized_query or "").strip().split()
    if not tokens:
        return "", [], []

    # Load synonyms map only if we will apply it.
    syn_map: Dict[str, Set[str]] = {}
    syn_source = "none"
    if apply_synonyms and _syn is not None:
        syn_map, syn_source = _load_synonyms_map(opts.synonyms_path)

    synonym_expansions: list[tuple[str, tuple[str, ...], str]] = []
    morphology_expansions: list[tuple[str, tuple[str, ...], str]] = []

    # Prepare corpus lexicon set for filtering morphology variants (optional but recommended).
    # We keep this best-effort: if lexicon load fails, we still can expand, but we prefer
    # filtering when possible.
    lexicon_terms: Set[str] = set()
    if apply_morphology:
        try:
            from . import lexicon as _lexicon
        except Exception:
            _lexicon = None  # type: ignore[assignment]

        if _lexicon is not None:
            try:
                _lexicon.ensure_lexicon(
                    conn,
                    min_len=int(opts.morphology_min_token_len),
                    force_rebuild=False,
                )
                min_freq = max(2, int(getattr(opts, "typos_min_freq", 2)))
                terms = _lexicon.load_lexicon_terms(
                    conn, min_freq=min_freq, limit=None, order_by="freq_desc"
                )
                lexicon_terms = set(str(t) for t in terms)
            except Exception:
                lexicon_terms = set()

    # Morph analyzer (RU) is optional; instantiate once per expansion call
    ru_analyzer = None
    if _morph is not None and apply_morphology and bool(opts.morphology_ru_enabled):
        try:
            ru_analyzer = _morph.RussianMorphology()
        except Exception:
            ru_analyzer = None

    def _eligible_for_synonyms(t: str) -> bool:
        if _syn is not None:
            return _syn.is_ordinary_word(t, min_len=int(opts.synonyms_min_token_len))
        return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё]+", t)) and len(t) >= int(
            opts.synonyms_min_token_len
        )

    def _eligible_for_morphology(t: str) -> bool:
        if _morph is not None:
            return _morph.is_ordinary_word(
                t, min_len=int(opts.morphology_min_token_len)
            )
        return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё]+", t)) and len(t) >= int(
            opts.morphology_min_token_len
        )

    # Build OR-groups (token/phrase + expansions)
    groups: List[List[str]] = []
    i = 0
    include_phrases = bool(opts.synonyms_include_phrases)
    max_per = (
        int(opts.synonyms_max_per_term) if int(opts.synonyms_max_per_term) > 0 else 0
    )

    while i < len(tokens):
        # Phrase-aware synonyms (only if synonyms enabled and phrases allowed)
        phrase_group: Optional[List[str]] = None
        phrase_text: Optional[str] = None

        if apply_synonyms and include_phrases and syn_map and _syn is not None:
            for n in (3, 2):
                if i + n <= len(tokens):
                    phrase_raw = " ".join(tokens[i : i + n])
                    phrase_norm = _syn.normalize_term(phrase_raw)
                    if phrase_norm in syn_map:
                        syns = sorted(set(syn_map.get(phrase_norm, set()) or []))
                        if max_per:
                            syns = syns[:max_per]
                        phrase_group = [phrase_norm] + [
                            s for s in syns if s and s != phrase_norm
                        ]
                        phrase_text = phrase_raw
                        break

        if phrase_group and phrase_text:
            groups.append(phrase_group)
            added = tuple(phrase_group[1:])
            if added:
                synonym_expansions.append((phrase_text, added, syn_source))
            i += len(phrase_text.split())
            continue

        # Single token group
        raw = tokens[i]
        tok_norm = (
            _syn.normalize_term(raw) if _syn is not None else (raw.strip().casefold())
        )
        group_terms: List[str] = [tok_norm]
        syn_added: List[str] = []

        # Synonyms expansion
        if (
            apply_synonyms
            and syn_map
            and _syn is not None
            and _eligible_for_synonyms(raw)
        ):
            syns = sorted(set(syn_map.get(tok_norm, set()) or []))
            if syns:
                if max_per:
                    syns = syns[:max_per]
                for s in syns:
                    if s and s != tok_norm and s not in group_terms:
                        group_terms.append(s)
                        syn_added.append(s)

        if syn_added:
            synonym_expansions.append((raw, tuple(syn_added), syn_source))

        # Morphology expansion (lexicon-filtered)
        if _morph is not None and apply_morphology and _eligible_for_morphology(raw):
            try:
                mo = _morph.MorphologyOptions(
                    enabled=True,
                    ru_enabled=bool(opts.morphology_ru_enabled),
                    ru_min_len=int(opts.morphology_min_token_len),
                    ru_max_variants=int(opts.morphology_ru_max_variants),
                    en_enabled=bool(opts.morphology_en_enabled),
                    en_min_len=int(opts.morphology_min_token_len),
                    en_max_variants=int(opts.morphology_en_max_variants),
                    preserve_case=True,
                )
                variants = _morph.expand_token(raw, options=mo, ru=ru_analyzer)
            except Exception:
                variants = []

            if variants:
                added_vars: List[str] = []
                syn_added_from_morph: List[str] = []

                for mv in variants:
                    vnorm = (
                        _syn.normalize_term(mv.variant)
                        if _syn is not None
                        else (mv.variant.strip().casefold())
                    )
                    if not vnorm or vnorm in group_terms:
                        continue

                    # Filter variants by corpus lexicon if available (best-effort).
                    #
                    # If the variant itself is not present in the corpus, we normally skip it
                    # to avoid widening the query with forms that never occur in indexed text.
                    #
                    # IMPORTANT: still allow the variant to contribute its SYNONYMS even when the
                    # variant isn't in the corpus lexicon. This enables RU->EN cross-language search:
                    #   "напряжения" -> morph base "напряжение" (not in corpus) -> synonym "voltage" (in corpus)
                    if lexicon_terms and vnorm not in lexicon_terms:
                        if apply_synonyms and syn_map and _syn is not None:
                            syns2 = sorted(set(syn_map.get(vnorm, set()) or []))
                            if max_per:
                                syns2 = syns2[:max_per]
                            for s2 in syns2:
                                if s2 and s2 not in group_terms:
                                    group_terms.append(s2)
                                    syn_added_from_morph.append(s2)
                        continue

                    group_terms.append(vnorm)
                    added_vars.append(vnorm)

                    # Apply synonym expansion to morphology-derived variants as well.
                    # This matters for RU inflections:
                    #   "напряжения" -> morph includes "напряжение" -> synonyms for "напряжение" should apply.
                    if apply_synonyms and syn_map and _syn is not None:
                        syns2 = sorted(set(syn_map.get(vnorm, set()) or []))
                        if max_per:
                            syns2 = syns2[:max_per]
                        for s2 in syns2:
                            if s2 and s2 not in group_terms:
                                group_terms.append(s2)
                                syn_added_from_morph.append(s2)

                if added_vars:
                    method = "morph"
                    if any(
                        getattr(mv, "source", "").startswith("ru:") for mv in variants
                    ):
                        method = "ru:pymorphy3"
                    elif any(
                        getattr(mv, "source", "").startswith("en:") for mv in variants
                    ):
                        method = "en:heur"
                    morphology_expansions.append((raw, tuple(added_vars), method))

                if syn_added_from_morph:
                    # Record as a separate expansion note to keep explainability.
                    uniq = tuple(sorted(set(syn_added_from_morph)))
                    synonym_expansions.append(
                        (f"{raw} (via morphology)", uniq, syn_source)
                    )

        groups.append(group_terms)
        i += 1

    # Convert OR-groups to FTS5 query
    if _syn is not None:
        expanded_fts = _syn.build_simple_fts_query_from_groups(groups)
    else:
        # Minimal fallback: join tokens with AND (no OR support)
        expanded_fts = " AND ".join([t for t in tokens if t]).strip()

    return expanded_fts, synonym_expansions, morphology_expansions


def search(
    conn: sqlite3.Connection,
    user_query: str,
    *,
    options: Optional[SearchOptions] = None,
) -> tuple[QueryPlan, list[LineHit] | list[PageHit]]:
    """
    Run a query-time search against the SQLite FTS5 index.

    Implements:
      (1) Query formation (build_fts5_query)
      (4) FTS5 MATCH + BM25 ranking
      (2) Typo correction (optional, safe): only "ordinary words" (ru/en letters-only)

    Returns:
      (QueryPlan, hits)

    Notes:
      - For exact citations, prefer target="lines".
      - For broader context, use target="pages".
      - Typo correction is applied only for simple-mode queries (non-advanced FTS).
    """
    _ensure_schema_present(conn)

    opts = options or SearchOptions()
    limit = _require_positive_int("limit", opts.limit)
    offset = _require_positive_int("offset", opts.offset)

    def _run_once(fts_query: str) -> list[LineHit] | list[PageHit]:
        if opts.target == "lines":
            return _search_lines(
                conn,
                fts_query,
                limit=limit,
                offset=offset,
                snippet_start=opts.snippet_start,
                snippet_end=opts.snippet_end,
                snippet_ellipsis=opts.snippet_ellipsis,
                snippet_tokens=opts.snippet_tokens,
            )
        if opts.target == "pages":
            return _search_pages(
                conn,
                fts_query,
                limit=limit,
                offset=offset,
                snippet_start=opts.snippet_start,
                snippet_end=opts.snippet_end,
                snippet_ellipsis=opts.snippet_ellipsis,
                snippet_tokens=opts.snippet_tokens,
            )
        raise ValueError(f"Unsupported target: {opts.target!r}")

    if limit == 0:
        return build_fts5_query(user_query, mode=opts.query_mode), []

    plan = build_fts5_query(user_query, mode=opts.query_mode)
    if not plan.fts:
        raise SearchError("Empty query after normalization. Provide a non-empty query.")

    hits = _run_once(plan.fts)

    # We continue the pipeline by updating (plan_used, hits_used) instead of returning early.
    plan_used = plan
    hits_used = hits

    # -----------------------------
    # Step 2: Typo correction (safe RU/EN ordinary words only)
    # -----------------------------
    typos_mode = (opts.typos_mode or "fallback").strip().lower()
    typos_enabled = bool(opts.typos_enabled) and typos_mode != "off"

    # Only apply to non-advanced queries (simple tokenized queries).
    if typos_enabled and (not plan_used.is_advanced) and plan_used.mode == "simple":
        # In fallback mode, only apply if first run returned 0 hits.
        should_try_typos = typos_mode == "always" or (
            typos_mode == "fallback" and not hits_used
        )

        if should_try_typos:
            # If there are no ordinary tokens, don't build lexicon or attempt corrections.
            try:
                from . import typos as _typos
            except Exception as e:  # pragma: no cover
                raise SearchError(f"Typo correction module not available: {e}") from e

            tokens = _typos.tokenize_simple_query(plan_used.normalized)
            has_eligible = any(
                _typos.is_ordinary_word(t, min_len=int(opts.typos_min_token_len))
                for t in tokens
            )

            if has_eligible:
                # Ensure corpus lexicon exists (lazy build).
                try:
                    from . import lexicon as _lexicon
                except Exception as e:  # pragma: no cover
                    raise SearchError(f"Lexicon module not available: {e}") from e

                try:
                    _lexicon.ensure_lexicon(
                        conn,
                        min_len=int(opts.typos_min_token_len),
                        force_rebuild=bool(opts.typos_force_rebuild),
                    )
                    terms = _lexicon.load_lexicon_terms(
                        conn,
                        min_freq=int(opts.typos_min_freq),
                        limit=opts.typos_lexicon_limit,
                        order_by="freq_desc",
                    )
                except Exception as e:
                    raise SearchError(
                        f"Failed to build/load corpus lexicon for typo correction: {e}"
                    ) from e

                if terms:
                    lex = _typos.Lexicon(terms)
                    typo_opts = _typos.TypoOptions(
                        enabled=True,
                        mode=str(typos_mode),
                        min_token_len=int(opts.typos_min_token_len),
                        min_score=float(opts.typos_min_score),
                        max_edits=int(opts.typos_max_edits),
                        preserve_case=True,
                    )

                    corrected_normalized, corrections = _typos.correct_simple_query(
                        plan_used.normalized, lex, options=typo_opts
                    )

                    if corrections:
                        corrected_plan_tmp = build_fts5_query(
                            corrected_normalized, mode="simple"
                        )
                        corrected_plan = _query_plan_with_typos(
                            plan_used,
                            corrected_normalized=corrected_plan_tmp.normalized,
                            corrected_fts=corrected_plan_tmp.fts,
                            corrections=corrections,
                        )
                        plan_used = corrected_plan
                        hits_used = _run_once(plan_used.fts)

    # -----------------------------
    # Step 3: Synonyms / morphology expansion (simple mode only)
    # -----------------------------
    syn_mode = (opts.synonyms_mode or "always").strip().lower()
    morph_mode = (opts.morphology_mode or "always").strip().lower()
    syn_enabled = bool(opts.synonyms_enabled) and syn_mode != "off"
    morph_enabled = bool(opts.morphology_enabled) and morph_mode != "off"

    if (
        (syn_enabled or morph_enabled)
        and (not plan_used.is_advanced)
        and plan_used.mode == "simple"
    ):
        # Apply modes cleanly per component.
        # - always: expand regardless of existing hits
        # - fallback: expand only if there are 0 hits so far
        want_syn = syn_enabled and (
            syn_mode == "always" or (syn_mode == "fallback" and not hits_used)
        )
        want_morph = morph_enabled and (
            morph_mode == "always" or (morph_mode == "fallback" and not hits_used)
        )

        if want_syn or want_morph:
            expanded_fts, syn_applied, morph_applied = _expand_simple_query_fts(
                plan_used.normalized,
                conn=conn,
                opts=opts,
                apply_synonyms=bool(want_syn),
                apply_morphology=bool(want_morph),
            )

            expanded_fts = (expanded_fts or "").strip()
            if (
                expanded_fts
                and expanded_fts != plan_used.fts
                and (syn_applied or morph_applied)
            ):
                plan_used = _query_plan_with_expansions(
                    plan_used,
                    expanded_fts=expanded_fts,
                    synonym_expansions=syn_applied,
                    morphology_expansions=morph_applied,
                )
                hits_used = _run_once(plan_used.fts)

    return plan_used, hits_used


def _search_lines(
    conn: sqlite3.Connection,
    fts_query: str,
    *,
    limit: int,
    offset: int,
    snippet_start: str,
    snippet_end: str,
    snippet_ellipsis: str,
    snippet_tokens: int,
) -> list[LineHit]:
    sql = """
    SELECT
      d.rel_path                AS rel_path,
      p.page_no                 AS page_no,
      l.line_no                 AS line_no,
      l.text                    AS text,
      snippet(lines_fts, 0, ?, ?, ?, ?) AS snippet,
      bm25(lines_fts)           AS score,
      p.is_ocr                  AS is_ocr
    FROM lines_fts
    JOIN lines     AS l ON l.line_id = lines_fts.rowid
    JOIN pages     AS p ON p.page_id = l.page_id
    JOIN documents AS d ON d.doc_id  = p.doc_id
    WHERE lines_fts MATCH ?
    ORDER BY score ASC
    LIMIT ? OFFSET ?;
    """
    try:
        rows = conn.execute(
            sql,
            (
                str(snippet_start),
                str(snippet_end),
                str(snippet_ellipsis),
                int(snippet_tokens),
                str(fts_query),
                int(limit),
                int(offset),
            ),
        ).fetchall()
    except sqlite3.OperationalError as e:
        # Typical cases: "fts5: syntax error near ..."
        raise SearchError(
            "FTS query syntax error. If you used advanced operators/quotes, "
            "try simplifying the query. "
            f"Details: {e}"
        ) from e

    out: list[LineHit] = []
    for r in rows:
        out.append(
            LineHit(
                rel_path=str(r["rel_path"]),
                page_no=int(r["page_no"]),
                line_no=int(r["line_no"]),
                text=str(r["text"]),
                snippet=str(r["snippet"]),
                score=float(r["score"]),
                is_ocr=bool(int(r["is_ocr"])),
            )
        )
    return out


def _search_pages(
    conn: sqlite3.Connection,
    fts_query: str,
    *,
    limit: int,
    offset: int,
    snippet_start: str,
    snippet_end: str,
    snippet_ellipsis: str,
    snippet_tokens: int,
) -> list[PageHit]:
    sql = """
    SELECT
      d.rel_path AS rel_path,
      p.page_no  AS page_no,
      p.text_len AS text_len,
      snippet(pages_fts, 0, ?, ?, ?, ?) AS snippet,
      bm25(pages_fts) AS score,
      p.is_ocr AS is_ocr
    FROM pages_fts
    JOIN pages     AS p ON p.page_id = pages_fts.rowid
    JOIN documents AS d ON d.doc_id  = p.doc_id
    WHERE pages_fts MATCH ?
    ORDER BY score ASC
    LIMIT ? OFFSET ?;
    """
    try:
        rows = conn.execute(
            sql,
            (
                str(snippet_start),
                str(snippet_end),
                str(snippet_ellipsis),
                int(snippet_tokens),
                str(fts_query),
                int(limit),
                int(offset),
            ),
        ).fetchall()
    except sqlite3.OperationalError as e:
        raise SearchError(
            "FTS query syntax error. If you used advanced operators/quotes, "
            "try simplifying the query. "
            f"Details: {e}"
        ) from e

    out: list[PageHit] = []
    for r in rows:
        out.append(
            PageHit(
                rel_path=str(r["rel_path"]),
                page_no=int(r["page_no"]),
                text_len=int(r["text_len"]),
                snippet=str(r["snippet"]),
                score=float(r["score"]),
                is_ocr=bool(int(r["is_ocr"])),
            )
        )
    return out


# -----------------------------
# Optional helpers (useful now, and later for reranking/context)
# -----------------------------


def fetch_line_context(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    page_no: int,
    line_no: int,
    before: int = 2,
    after: int = 2,
) -> list[tuple[int, str]]:
    """
    Fetch a small window of (line_no, text) around a specific line.

    This is not required for MVP search, but is handy for nicer citations and
    will remain useful for later reranking/LLM prompting.

    The lookup is by (rel_path, page_no) -> page_id, then range query on lines.
    """
    before = _require_positive_int("before", before)
    after = _require_positive_int("after", after)

    row = conn.execute(
        """
        SELECT p.page_id
        FROM pages p
        JOIN documents d ON d.doc_id = p.doc_id
        WHERE d.rel_path = ? AND p.page_no = ?
        """,
        (str(rel_path), int(page_no)),
    ).fetchone()
    if row is None:
        return []

    page_id = int(row["page_id"])
    lo = max(1, int(line_no) - before)
    hi = int(line_no) + after

    rows = conn.execute(
        """
        SELECT line_no, text
        FROM lines
        WHERE page_id = ? AND line_no BETWEEN ? AND ?
        ORDER BY line_no
        """,
        (int(page_id), int(lo), int(hi)),
    ).fetchall()

    return [(int(r["line_no"]), str(r["text"])) for r in rows]
