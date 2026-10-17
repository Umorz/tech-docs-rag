from __future__ import annotations

"""
Morphology helpers for Query-time expansion (Указания.md шаг 3: "Синонимы / морфология").

This module is intentionally conservative and domain-safe for technical corpora:
- It only operates on "ordinary words" (letters-only tokens).
- It avoids touching part numbers / identifiers (LM2596, 3.3V, I/O, PB5, etc.).
- It is designed to be composed with:
    (1) query formation
    (2) typo correction
    (4) FTS5 MATCH / BM25
  and later with (5) embeddings and (6) reranking.

Russian morphology uses `pymorphy3` when installed (optional dependency), with a conservative
heuristic fallback when it is unavailable.
English morphology is a lightweight heuristic expansion (stdlib only).

Typical usage pattern (in query pipeline):
1) tokenize simple query (whitespace)
2) for each token:
     - if RU: expand via pymorphy3 lexeme (capped)
     - if EN: expand via conservative suffix rules (capped)
3) incorporate expansions into the FTS query with OR-groups:
     token -> (token OR v1 OR v2 ...)
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, List, Literal, Optional, Sequence, Set

Lang = Literal["ru", "en", "other", "mixed"]


# -----------------------------
# Optional dependency: pymorphy3
# -----------------------------

try:  # pragma: no cover (environment-dependent)
    import pymorphy3  # type: ignore
except Exception:  # pragma: no cover
    pymorphy3 = None  # type: ignore[assignment]


# -----------------------------
# Token eligibility / language detection
# -----------------------------

# Letters-only tokens (Latin or Cyrillic incl. Ёё). No digits/underscore/hyphen/punct.
_LETTERS_ONLY_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+$")

_HAS_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_HAS_LATIN_RE = re.compile(r"[A-Za-z]")


def is_ordinary_word(token: str, *, min_len: int = 4) -> bool:
    """
    True if token is eligible for morphology expansion:
    - letters-only (Latin/Cyrillic)
    - length >= min_len
    """
    t = (token or "").strip()
    if len(t) < int(min_len):
        return False
    return bool(_LETTERS_ONLY_RE.fullmatch(t))


def detect_lang(token: str) -> Lang:
    """
    Detect whether token looks RU, EN, mixed or other.

    We only support pure Cyrillic or pure Latin tokens here.
    Mixed tokens (e.g., "ATmega", "USBпорт") return "mixed" and should be skipped.
    """
    t = (token or "").strip()
    if not t:
        return "other"

    has_ru = bool(_HAS_CYRILLIC_RE.search(t))
    has_en = bool(_HAS_LATIN_RE.search(t))

    if has_ru and has_en:
        return "mixed"
    if has_ru:
        return "ru"
    if has_en:
        return "en"
    return "other"


def _casefold_key(s: str) -> str:
    return (s or "").casefold()


def _apply_original_casing(original: str, variant: str) -> str:
    """
    Apply a minimal casing pattern from `original` to `variant`.

    Patterns:
      - ALL CAPS   -> variant.upper()
      - TitleCase  -> Variant (first char upper, rest as-is)
      - otherwise  -> variant (unchanged)
    """
    if not variant:
        return variant
    if not original:
        return variant

    if original.isupper():
        return variant.upper()
    if len(original) >= 2 and original[0].isupper() and original[1:].islower():
        return variant[0].upper() + variant[1:]
    return variant


# -----------------------------
# Options / results
# -----------------------------


@dataclass(frozen=True)
class MorphologyOptions:
    enabled: bool = True

    # RU
    ru_enabled: bool = True
    ru_min_len: int = 4
    ru_max_variants: int = 10  # cap to avoid huge OR queries
    ru_include_normal_form: bool = True
    ru_include_lexeme: bool = True  # includes grammatical variants (lexeme)

    # EN (heuristic)
    en_enabled: bool = True
    en_min_len: int = 4
    en_max_variants: int = 6  # very conservative
    en_include_singular_plural: bool = True
    en_include_basic_suffix_strips: bool = True

    # Global
    preserve_case: bool = True


@dataclass(frozen=True)
class MorphVariant:
    """
    Explainable variant for a token.

    `source` describes where the variant came from:
      - "ru:normal"   (pymorphy3 normal_form)
      - "ru:lexeme"   (pymorphy3 lexeme form)
      - "en:heur"     (english heuristic)
    """

    token: str
    variant: str
    source: str


# -----------------------------
# Russian morphology (pymorphy3 + heuristic fallback)
# -----------------------------


# Conservative RU ending stripping. We only generate a small set of likely base forms.
# This is NOT a full morphological analyzer, but it helps recover common inflections.
_RU_COMMON_ENDINGS: tuple[str, ...] = (
    # longer first
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ам",
    "ям",
    "ом",
    "ем",
    "ой",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ым",
    "им",
    "ую",
    "юю",
    "а",
    "я",
    "ы",
    "и",
    "е",
    "у",
    "ю",
    "о",
)

# Heuristic suffix->suffix rewrites to recover common noun lemmas like:
#   напряжения -> напряжение
#   измерения  -> измерение
#   описания   -> описание (for "-ание" words)
#
# We keep this list small and conservative on purpose.
_RU_SUFFIX_REWRITES: tuple[tuple[str, str], ...] = (
    ("ением", "ение"),
    ("ении", "ение"),
    ("ению", "ение"),
    ("ения", "ение"),
    ("анием", "ание"),
    ("ании", "ание"),
    ("анию", "ание"),
    ("ания", "ание"),
)


def _russian_heuristic_bases(token: str, *, min_len: int) -> Iterator[str]:
    """
    Yield conservative "base" candidates for a Russian word by stripping common endings.

    We only return letters-only candidates and do not attempt to generate full paradigms.
    """
    t = (token or "").strip()
    if not t:
        return

    # Work in casefold for stability; caller can re-apply casing.
    low = t.casefold()

    # First, try conservative suffix rewrites for common "-ение"/"-ание" nouns.
    # This specifically addresses cases like:
    #   "напряжения" -> "напряжение"
    # where naive stripping "я" would produce "напряжени".
    for src_suf, dst_suf in _RU_SUFFIX_REWRITES:
        if low.endswith(src_suf):
            stem = low[: -len(src_suf)]
            cand = stem + dst_suf
            if len(cand) >= int(min_len) and is_ordinary_word(
                cand, min_len=int(min_len)
            ):
                yield cand

    # Always consider stripping a trailing soft sign (very common in oblique forms).
    if low.endswith("ь") and len(low) - 1 >= int(min_len):
        cand = low[:-1]
        if is_ordinary_word(cand, min_len=int(min_len)):
            yield cand

    # Generic ending stripping (may produce stems; keep conservative).
    for suf in _RU_COMMON_ENDINGS:
        if low.endswith(suf) and len(low) - len(suf) >= int(min_len):
            cand = low[: -len(suf)]
            if is_ordinary_word(cand, min_len=int(min_len)):
                yield cand


def _expand_russian_heuristic(
    token: str, *, options: MorphologyOptions
) -> List[MorphVariant]:
    """
    Heuristic RU morphology expansion used when pymorphy3 is unavailable.

    Returns only a few base-form candidates to improve recall without blowing up queries.
    """
    if not options.enabled or not options.ru_enabled:
        return []
    if not is_ordinary_word(token, min_len=int(options.ru_min_len)):
        return []
    if detect_lang(token) != "ru":
        return []

    out: List[MorphVariant] = []
    seen: Set[str] = set()

    for cand in _russian_heuristic_bases(token, min_len=int(options.ru_min_len)):
        key = _casefold_key(cand)
        if key in seen:
            continue
        seen.add(key)

        v = cand
        if options.preserve_case:
            v = _apply_original_casing(token, v)

        out.append(MorphVariant(token=token, variant=v, source="ru:heur"))
        if len(out) >= int(options.ru_max_variants):
            break

    return out


class RussianMorphology:
    """
    Russian morphology expander using pymorphy3 (if available), otherwise a conservative
    heuristic fallback.

    Safe defaults:
      - only expands letters-only Cyrillic words
      - caps number of variants
    """

    def __init__(self) -> None:
        self._analyzer = None
        if pymorphy3 is not None:  # pragma: no cover (env-dependent)
            self._analyzer = pymorphy3.MorphAnalyzer()

    @property
    def available(self) -> bool:
        return self._analyzer is not None

    @lru_cache(maxsize=50_000)
    def _parse_best(self, token_casefold: str):
        if self._analyzer is None:
            return None
        # pymorphy3 expects original-ish token; we pass casefolded for cache stability.
        parses = self._analyzer.parse(token_casefold)
        if not parses:
            return None
        return parses[0]

    def expand(
        self,
        token: str,
        *,
        options: MorphologyOptions,
    ) -> List[MorphVariant]:
        if not options.enabled or not options.ru_enabled:
            return []

        if not is_ordinary_word(token, min_len=int(options.ru_min_len)):
            return []
        if detect_lang(token) != "ru":
            return []
        if self._analyzer is None:
            return _expand_russian_heuristic(token, options=options)

        key = _casefold_key(token)
        p = self._parse_best(key)
        if p is None:
            return []

        out: List[MorphVariant] = []
        seen: Set[str] = set()

        def _add(v: str, source: str) -> None:
            v = (v or "").strip()
            if not v:
                return
            if not is_ordinary_word(v, min_len=int(options.ru_min_len)):
                return
            vv = _casefold_key(v)
            if vv in seen:
                return
            seen.add(vv)
            if options.preserve_case:
                v = _apply_original_casing(token, v)
            out.append(MorphVariant(token=token, variant=v, source=source))

        # Normal form (lemma)
        if options.ru_include_normal_form:
            try:
                _add(str(p.normal_form), "ru:normal")
            except Exception:
                pass

        # Lexeme variants (all forms); may be large -> cap.
        if options.ru_include_lexeme:
            try:
                for form in p.lexeme:
                    if len(out) >= int(options.ru_max_variants):
                        break
                    _add(str(getattr(form, "word", "")), "ru:lexeme")
            except Exception:
                pass

        # Ensure original token is represented by caller (we typically don't include it here).
        return out[: int(options.ru_max_variants)]


# -----------------------------
# English morphology (heuristic)
# -----------------------------


def _english_heuristic_variants(
    token: str, *, options: MorphologyOptions
) -> Iterator[str]:
    """
    Generate conservative English variants.

    Philosophy: prefer precision and safety.
    We only do tiny transformations that often help:
      - singular/plural forms
      - stripping common suffixes (s/es/ed/ing) to reach a potential base
    """
    t = (token or "").strip()
    if not t:
        return
    if not is_ordinary_word(t, min_len=int(options.en_min_len)):
        return
    if detect_lang(t) != "en":
        return

    base = t

    # Basic singularization
    if options.en_include_singular_plural:
        low = base.lower()
        if low.endswith("ies") and len(base) > 4:
            yield base[:-3] + "y"
        if low.endswith("es") and len(base) > 3:
            # boxes -> box, classes -> class (approx)
            yield base[:-2]
        if low.endswith("s") and len(base) > 3:
            yield base[:-1]

        # Basic pluralization
        if not low.endswith("s") and len(base) > 2:
            if low.endswith(("s", "x", "z", "ch", "sh")):
                yield base + "es"
            elif low.endswith("y") and len(base) > 3 and low[-2] not in "aeiou":
                yield base[:-1] + "ies"
            else:
                yield base + "s"

    # Strip common verb suffixes
    if options.en_include_basic_suffix_strips:
        low = base.lower()
        if low.endswith("ing") and len(base) > 5:
            yield base[:-3]
            # "making" -> "make" (naive)
            if base[-4].lower() not in "aeiou" and len(base) > 6:
                yield base[:-3] + "e"
        if low.endswith("ed") and len(base) > 4:
            yield base[:-2]
            if base[-3].lower() == base[-4].lower() and len(base) > 5:
                # "stopped" -> "stop" (naive)
                yield base[:-3]
        if low.endswith("er") and len(base) > 4:
            yield base[:-2]
        if low.endswith("ers") and len(base) > 5:
            yield base[:-3]


def expand_english(
    token: str,
    *,
    options: MorphologyOptions,
) -> List[MorphVariant]:
    if not options.enabled or not options.en_enabled:
        return []
    if not is_ordinary_word(token, min_len=int(options.en_min_len)):
        return []
    if detect_lang(token) != "en":
        return []

    out: List[MorphVariant] = []
    seen: Set[str] = set()

    def _add(v: str) -> None:
        v = (v or "").strip()
        if not v:
            return
        if not is_ordinary_word(v, min_len=int(options.en_min_len)):
            return
        key = _casefold_key(v)
        if key in seen:
            return
        seen.add(key)
        if options.preserve_case:
            v = _apply_original_casing(token, v)
        out.append(MorphVariant(token=token, variant=v, source="en:heur"))

    for v in _english_heuristic_variants(token, options=options):
        if len(out) >= int(options.en_max_variants):
            break
        _add(v)

    return out[: int(options.en_max_variants)]


# -----------------------------
# High-level API
# -----------------------------


def expand_token(
    token: str,
    *,
    options: Optional[MorphologyOptions] = None,
    ru: Optional[RussianMorphology] = None,
) -> List[MorphVariant]:
    """
    Expand a single token into morphology variants (RU via pymorphy3, EN heuristics).

    Returns variants ONLY (caller should keep the original token separately).
    """
    opts = options or MorphologyOptions()
    if not opts.enabled:
        return []

    lang = detect_lang(token)
    if lang == "mixed" or lang == "other":
        return []

    variants: List[MorphVariant] = []

    if lang == "ru" and opts.ru_enabled:
        analyzer = ru or RussianMorphology()
        variants.extend(analyzer.expand(token, options=opts))

    if lang == "en" and opts.en_enabled:
        variants.extend(expand_english(token, options=opts))

    # Deduplicate by casefolded variant text, preserve order
    seen: Set[str] = set()
    deduped: List[MorphVariant] = []
    for mv in variants:
        k = _casefold_key(mv.variant)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(mv)

    # Apply caps again (defensive)
    if lang == "ru":
        return deduped[: int(opts.ru_max_variants)]
    if lang == "en":
        return deduped[: int(opts.en_max_variants)]
    return deduped


def expand_tokens(
    tokens: Sequence[str],
    *,
    options: Optional[MorphologyOptions] = None,
    ru: Optional[RussianMorphology] = None,
) -> List[MorphVariant]:
    """
    Expand a list of tokens and return all variants (excluding originals).
    """
    out: List[MorphVariant] = []
    for t in tokens:
        out.extend(expand_token(t, options=options, ru=ru))
    return out
