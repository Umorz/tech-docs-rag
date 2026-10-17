from __future__ import annotations

"""
Typo correction (Query-time step 2) for "ordinary words" in RU/EN queries.

Goals
-----
- Help users when they misspell common words (ru/en) in a query.
- Be safe for technical corpora: DO NOT touch part numbers, signal names, voltages, etc.
- Work as a small, clean layer that can be composed with:
    - query formation (step 1)
    - FTS5 MATCH + BM25 (step 4)
    - future: synonyms/morphology/embeddings/reranking

Design principles
-----------------
1) Conservative token eligibility:
   - correct ONLY tokens that are letters-only (Cyrillic/Latin) and have a minimum length
   - never correct tokens with digits or punctuation (e.g. "LM2596", "3.3V", "I/O", "PB5")

2) Corpus-driven lexicon:
   - correction must be against a lexicon derived from the indexed corpus, not a generic dictionary

3) Pluggable scoring:
   - prefer `rapidfuzz` when installed (fast & good)
   - fall back to stdlib `difflib` otherwise

4) Explainability:
   - return structured corrections so the caller can show "volatge -> voltage"

This module intentionally does NOT:
- build the lexicon (that belongs to a separate layer that knows about SQLite / corpus)
- parse advanced FTS queries (quotes/parentheses/AND/OR). Apply typo correction in
  "simple query mode" (tokenized by spaces) or in a caller-controlled manner.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Sequence

# Optional dependency (preferred)
try:
    from rapidfuzz import fuzz, process  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None  # type: ignore[assignment]
    process = None  # type: ignore[assignment]

import difflib

# -----------------------------
# Public models / options
# -----------------------------


@dataclass(frozen=True)
class TypoOptions:
    """
    Configuration for typo correction.

    mode:
      - "fallback": only apply corrections if initial retrieval returned 0 hits
      - "always": always apply corrections (NOT recommended for technical domains)
    """

    enabled: bool = True
    mode: str = "fallback"

    # Token eligibility
    min_token_len: int = 4

    # Candidate selection thresholds
    min_score: float = (
        85.0  # 0..100 similarity score threshold (allows common transpositions)
    )
    max_edits: int = 2  # safety guard to avoid aggressive corrections

    # How many candidates to consider internally (only used by some scorers)
    max_candidates: int = 50

    # Case handling
    preserve_case: bool = True


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    score: float
    method: str  # "rapidfuzz" | "difflib"
    edits: Optional[int] = None


class TyposError(RuntimeError):
    """Raised when typo correction cannot be performed due to invalid inputs."""


# -----------------------------
# Token eligibility
# -----------------------------


# Letters-only token: Latin + Cyrillic (incl. Ёё) only.
# We intentionally do not accept underscore/digits/hyphen/etc.
_LETTERS_ONLY_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+$")


def is_ordinary_word(token: str, *, min_len: int = 4) -> bool:
    """
    Return True if token is safe to consider for spelling correction.

    Safe = letters-only (ru/en) and long enough.
    """
    token = (token or "").strip()
    if len(token) < int(min_len):
        return False
    return bool(_LETTERS_ONLY_RE.fullmatch(token))


def _casefold_key(s: str) -> str:
    # casefold handles more Unicode cases than lower()
    return (s or "").casefold()


def _apply_original_casing(original: str, corrected: str) -> str:
    """
    Apply a simple casing pattern from `original` to `corrected`.

    Patterns:
      - ALL CAPS      -> corrected.upper()
      - Title Case    -> corrected.capitalize()
      - otherwise     -> corrected as-is
    """
    if not corrected:
        return corrected
    if not original:
        return corrected

    if original.isupper():
        return corrected.upper()
    if len(original) >= 2 and original[0].isupper() and original[1:].islower():
        return corrected[0].upper() + corrected[1:]
    return corrected


# -----------------------------
# Lexicon adapters
# -----------------------------


class Lexicon:
    """
    A lightweight view over a corpus lexicon.

    You can pass:
      - a set[str]
      - a list/tuple[str]
      - a dict[str, int] where keys are terms (freq unused here but can help later)

    This wrapper normalizes membership checks via casefolding.
    """

    def __init__(self, terms: Iterable[str] | Mapping[str, int]):
        if isinstance(terms, Mapping):
            self._terms = list(terms.keys())
            self._term_set = {_casefold_key(t) for t in terms.keys()}
        else:
            self._terms = list(terms)
            self._term_set = {_casefold_key(t) for t in self._terms}

    def __contains__(self, term: str) -> bool:
        return _casefold_key(term) in self._term_set

    def iter_terms(self) -> Iterator[str]:
        yield from self._terms

    def __len__(self) -> int:
        return len(self._terms)


# -----------------------------
# Edit distance (safety guard)
# -----------------------------


def levenshtein_distance(a: str, b: str, *, max_edits: Optional[int] = None) -> int:
    """
    Compute Levenshtein edit distance with optional early exit (max_edits).

    This is used as a conservative guardrail, not as the primary scorer.
    """
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # If max_edits is set, a length gap already exceeding it cannot pass.
    if max_edits is not None and abs(len(a) - len(b)) > int(max_edits):
        return int(max_edits) + 1

    # Ensure a is the shorter string for memory efficiency.
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, ch_b in enumerate(b, start=1):
        curr = [j]
        min_in_row = curr[0]
        for i, ch_a in enumerate(a, start=1):
            ins = curr[i - 1] + 1
            dele = prev[i] + 1
            sub = prev[i - 1] + (0 if ch_a == ch_b else 1)
            v = ins if ins < dele else dele
            if sub < v:
                v = sub
            curr.append(v)
            if v < min_in_row:
                min_in_row = v

        prev = curr

        if max_edits is not None and min_in_row > int(max_edits):
            return int(max_edits) + 1

    return prev[-1]


# -----------------------------
# Candidate selection
# -----------------------------


def _best_match_rapidfuzz(
    token: str, lex: Lexicon, *, max_candidates: int
) -> Optional[tuple[str, float]]:
    if process is None or fuzz is None:
        return None

    # Rapidfuzz expects a sequence; we provide the raw terms.
    # WRatio is robust to minor differences.
    match = process.extractOne(
        token,
        lex.iter_terms(),
        scorer=fuzz.WRatio,
        score_cutoff=0,
        processor=None,  # don't pre-process; token already normalized by caller
    )
    if match is None:
        return None
    cand, score, _idx = match
    return str(cand), float(score)


def _best_match_difflib(token: str, lex: Lexicon) -> Optional[tuple[str, float]]:
    """
    Fallback matching using stdlib.

    Returns (candidate, score_0_100).
    """
    # difflib works best on smaller candidate sets; but we keep it as a fallback.
    # Using get_close_matches will internally use SequenceMatcher ratio.
    candidates = difflib.get_close_matches(
        token, list(lex.iter_terms()), n=1, cutoff=0.0
    )
    if not candidates:
        return None
    cand = candidates[0]
    score = difflib.SequenceMatcher(None, token, cand).ratio() * 100.0
    return str(cand), float(score)


def suggest_correction(
    token: str,
    lexicon: Lexicon,
    *,
    options: TypoOptions,
) -> Optional[Correction]:
    """
    Suggest a single correction for `token`, or return None if no safe correction found.

    The caller should ensure token eligibility via `is_ordinary_word()`.
    """
    if not options.enabled:
        return None

    if token in lexicon:
        return None

    # Prefer rapidfuzz if installed
    method = "difflib"
    best: Optional[tuple[str, float]] = None

    if process is not None and fuzz is not None:
        best = _best_match_rapidfuzz(
            token, lexicon, max_candidates=int(options.max_candidates)
        )
        method = "rapidfuzz"
    if best is None:
        best = _best_match_difflib(token, lexicon)
        method = "difflib"

    if best is None:
        return None

    cand, score = best
    if score < float(options.min_score):
        return None

    edits = None
    if options.max_edits is not None:
        edits = levenshtein_distance(
            _casefold_key(token), _casefold_key(cand), max_edits=int(options.max_edits)
        )
        if edits > int(options.max_edits):
            return None

    corrected = cand
    if options.preserve_case:
        corrected = _apply_original_casing(token, corrected)

    if _casefold_key(corrected) == _casefold_key(token):
        return None

    return Correction(
        original=token,
        corrected=corrected,
        score=float(score),
        method=method,
        edits=int(edits) if edits is not None else None,
    )


# -----------------------------
# Query-level helpers (simple mode)
# -----------------------------


def tokenize_simple_query(normalized_query: str) -> list[str]:
    """
    Tokenize a "simple mode" query by whitespace.
    """
    q = (normalized_query or "").strip()
    q = re.sub(r"\s+", " ", q).strip()
    return [t for t in q.split(" ") if t] if q else []


def correct_simple_query_tokens(
    tokens: Sequence[str],
    lexicon: Lexicon,
    *,
    options: TypoOptions,
) -> tuple[list[str], list[Correction]]:
    """
    Correct eligible tokens in a simple token list.

    Only corrects "ordinary words" (ru/en letters-only, length >= min_token_len).
    """
    out: list[str] = []
    corrections: list[Correction] = []

    for t in tokens:
        if not is_ordinary_word(t, min_len=int(options.min_token_len)):
            out.append(t)
            continue

        corr = suggest_correction(t, lexicon, options=options)
        if corr is None:
            out.append(t)
            continue

        out.append(corr.corrected)
        corrections.append(corr)

    return out, corrections


def correct_simple_query(
    normalized_query: str,
    lexicon: Lexicon,
    *,
    options: TypoOptions,
) -> tuple[str, list[Correction]]:
    """
    Apply typo correction to a normalized query string in "simple mode".

    Returns:
      (corrected_query, corrections)
    """
    tokens = tokenize_simple_query(normalized_query)
    corrected_tokens, corrections = correct_simple_query_tokens(
        tokens, lexicon, options=options
    )
    return " ".join(corrected_tokens).strip(), corrections
