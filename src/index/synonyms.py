from __future__ import annotations

"""
Query-time synonyms expansion (Step 3: "Синонимы / морфология").

This module focuses on *synonyms* expansion in a conservative, explainable way.
Russian/English morphology is intentionally NOT implemented here (it usually needs
a dedicated morphological analyzer and careful domain rules). We keep the design
compatible with adding morphology later.

Design goals
------------
- Keep project structure clean: small, dependency-free utility module.
- Be safe for technical corpora: expand only "ordinary words" (letters-only RU/EN)
  by default, so we don't mess with part numbers / identifiers.
- Be explainable: return structured info on what expansions were applied.
- Be composable with:
    Step 1: query formation
    Step 2: typo correction
    Step 4: FTS5 MATCH + BM25
    Future: embeddings / reranking

Where this fits
---------------
Callers should apply synonyms expansion ONLY in "simple" query mode (tokenized by
whitespace) and typically:
  normalize -> (optional typos) -> synonyms expansion -> build FTS query -> search

Storage format
--------------
This module supports loading synonyms from JSON in three compatible shapes:

1) Mapping form (simple dictionary):
   {
     "regulator": ["stabilizer", "voltage regulator"],
     "напряжение": ["вольтаж", "voltage"]
   }

2) Groups form (explicit symmetric groups):
   [
     ["voltage", "вольтаж", "напряжение"],
     ["capacitor", "cap", "конденсатор"]
   ]

3) Structured file with metadata + groups (recommended for larger projects):
   {
     "schema_version": 1,
     "groups": [
       { "id": "voltage", "terms": ["voltage", "напряжение", "вольтаж"], "tags": ["units"] },
       { "id": "capacitor", "terms": ["capacitor", "cap", "конденсатор"], "tags": ["passive"] }
     ]
   }

Internally we build a symmetric map:
  term -> set(other_terms)

Normalization
-------------
All terms are normalized with:
  - Unicode normalization via text_clean.normalize_unicode
  - casefold()
We store and match on this normalized form.

FTS5 query building
-------------------
We provide helpers to build a conservative FTS5 expression for simple mode:
- Each token becomes either:
    token
  or
    (token OR syn1 OR "multi word syn")
- The caller can then join tokens with AND (as you already do).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from . import text_clean

# -----------------------------
# Models / options
# -----------------------------


@dataclass(frozen=True)
class SynonymOptions:
    enabled: bool = True

    # Safety: expand only "ordinary words" by default (letters-only ru/en).
    ordinary_words_only: bool = True
    min_token_len: int = 4

    # Limit how much we expand a single term (avoid huge OR clauses).
    max_synonyms_per_term: int = 6

    # Whether to include multi-word phrases in expansions.
    include_phrases: bool = True

    # If True, will expand transitively within a group (A->B->C). Since we build
    # symmetric groups, this is effectively always True, but kept as a knob for
    # future changes.
    transitive: bool = True


@dataclass(frozen=True)
class AppliedSynonyms:
    token: str
    normalized_token: str
    added: Tuple[str, ...]  # normalized added terms
    source: str  # e.g. "file:data/synonyms.json" or "inline"
    notes: str = ""


@dataclass(frozen=True)
class ExpansionResult:
    """
    Expansion result for a token list.

    groups:
      list of term-lists (each group is OR'ed together)
      Example:
        tokens ["thermal", "resistance"] -> groups [["thermal"], ["resistance", "rth"]]
    applied:
      structured info on what changed
    """

    groups: List[List[str]]
    applied: List[AppliedSynonyms]


# -----------------------------
# Token eligibility (safety)
# -----------------------------


# Letters-only token: Latin + Cyrillic (incl. Ёё) only.
_LETTERS_ONLY_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+$")


def is_ordinary_word(token: str, *, min_len: int = 4) -> bool:
    token = (token or "").strip()
    if len(token) < int(min_len):
        return False
    return bool(_LETTERS_ONLY_RE.fullmatch(token))


# -----------------------------
# Normalization
# -----------------------------


def normalize_term(term: str) -> str:
    t = text_clean.normalize_unicode(term or "")
    t = t.strip().casefold()
    # Collapse internal whitespace for phrase stability
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _escape_fts_phrase(s: str) -> str:
    # Escape for use inside double quotes in SQLite FTS5.
    return (s or "").replace('"', '""')


def _needs_fts_quotes(term: str) -> bool:
    # Quote if contains whitespace or characters that are not "word" chars.
    # For safety we quote phrases and terms with punctuation.
    return bool(re.search(r"\s", term)) or not bool(
        re.fullmatch(r"\w+", term, flags=re.UNICODE)
    )


# -----------------------------
# Synonym map construction / loading
# -----------------------------


SynonymSource = Union[Mapping[str, Sequence[str]], Sequence[Sequence[str]]]


def build_synonym_map(source: SynonymSource) -> Dict[str, Set[str]]:
    """
    Build a symmetric synonym map from either mapping-form or groups-form input.

    Returns:
      dict[term] -> set(other_terms)
    All keys/values are normalized.
    """
    m: Dict[str, Set[str]] = {}

    def _add_pair(a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        m.setdefault(a, set()).add(b)

    if isinstance(source, Mapping):
        for k, vals in source.items():
            a = normalize_term(k)
            if not a:
                continue
            for v in vals or []:
                b = normalize_term(str(v))
                if not b:
                    continue
                _add_pair(a, b)
                _add_pair(b, a)
        return m

    # groups-form: [["a","b","c"], ["x","y"]]
    for group in source:
        terms = [normalize_term(str(x)) for x in (group or [])]
        terms = [t for t in terms if t]
        if len(terms) < 2:
            continue
        for i, a in enumerate(terms):
            for j, b in enumerate(terms):
                if i == j:
                    continue
                _add_pair(a, b)

    return m


def load_synonyms_json(path: str | Path) -> Dict[str, Set[str]]:
    """
    Load synonyms from a JSON file.

    Supported shapes (see module docstring):
      - mapping-form: { "term": ["syn1", "syn2"] }
      - groups-form:  [ ["a","b"], ["x","y","z"] ]
      - structured:   { "schema_version": 1, "groups": [ { "terms": [...] }, ... ] }

    Backward compatibility:
      - If a JSON object contains non-synonym metadata keys, they are ignored unless
        it matches the structured format ("groups") or an explicit "map" key.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # 1) groups-form (legacy): [["a","b"], ["x","y"]]
    if isinstance(data, list):
        return build_synonym_map([[str(x) for x in group] for group in data])

    if not isinstance(data, dict):
        raise ValueError(
            f"Unsupported synonyms JSON structure in {p}: expected object or array"
        )

    # 2) structured format with metadata + groups
    if "groups" in data:
        groups = data.get("groups") or []
        group_terms: List[List[str]] = []
        if isinstance(groups, list):
            for g in groups:
                terms = None
                if isinstance(g, dict):
                    terms = g.get("terms")
                elif isinstance(g, list):
                    terms = g
                if not isinstance(terms, list):
                    continue
                group_terms.append([str(x) for x in terms])
        return build_synonym_map(group_terms)

    # 3) explicit mapping under "map" key (optional convenience)
    if "map" in data and isinstance(data.get("map"), dict):
        mp = {str(k): [str(x) for x in (v or [])] for k, v in data["map"].items()}
        return build_synonym_map(mp)

    # 4) mapping-form fallback: keep only key->list entries (ignore metadata keys)
    mp2: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (list, tuple)):
            mp2[str(k)] = [str(x) for x in v]

    if mp2:
        return build_synonym_map(mp2)

    raise ValueError(
        f"Unsupported synonyms JSON structure in {p}: expected mapping, groups, or structured 'groups'"
    )


def merge_synonym_maps(*maps: Mapping[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """
    Merge multiple synonym maps into one symmetric map.

    Input maps may be:
      - term -> iterable of synonyms (normalized or not; we normalize here)
    """
    out: Dict[str, Set[str]] = {}
    for mp in maps:
        for k, vals in mp.items():
            a = normalize_term(str(k))
            if not a:
                continue
            for v in vals:
                b = normalize_term(str(v))
                if not b:
                    continue
                if a == b:
                    continue
                out.setdefault(a, set()).add(b)
                out.setdefault(b, set()).add(a)
    return out


# -----------------------------
# Expansion
# -----------------------------


def expand_tokens(
    tokens: Sequence[str],
    synonyms: Mapping[str, Set[str]],
    *,
    options: SynonymOptions,
    source_label: str = "inline",
) -> ExpansionResult:
    """
    Expand a list of *normalized* tokens using a synonym map.

    This function assumes tokens are already normalized (casefolded, unicode-normalized)
    OR at least consistent with normalize_term(). It will still normalize each token
    defensively.

    Returns:
      ExpansionResult:
        groups: list[list[str]] (each inner list is an OR group)
        applied: list of AppliedSynonyms
    """
    groups: List[List[str]] = []
    applied: List[AppliedSynonyms] = []

    for raw in tokens:
        norm = normalize_term(raw)
        if not norm:
            continue

        eligible = True
        if options.ordinary_words_only:
            eligible = is_ordinary_word(norm, min_len=int(options.min_token_len))

        if not options.enabled or not eligible:
            groups.append([norm])
            continue

        syns = list(synonyms.get(norm, set()) or [])
        if not syns:
            groups.append([norm])
            continue

        # Optionally drop phrases
        if not options.include_phrases:
            syns = [s for s in syns if " " not in s]

        # Deterministic ordering (helps debugging / stable plans)
        syns = sorted(set(syns))

        # Limit expansions to avoid query blow-ups
        if (
            options.max_synonyms_per_term is not None
            and options.max_synonyms_per_term > 0
        ):
            syns = syns[: int(options.max_synonyms_per_term)]

        # Build OR group with the original term first
        group = [norm] + [s for s in syns if s and s != norm]
        groups.append(group)

        added = tuple(s for s in group[1:] if s)
        if added:
            applied.append(
                AppliedSynonyms(
                    token=str(raw),
                    normalized_token=norm,
                    added=added,
                    source=str(source_label),
                    notes="",
                )
            )

    return ExpansionResult(groups=groups, applied=applied)


def build_fts_or_group(terms: Sequence[str]) -> str:
    """
    Build an FTS5 OR group for a list of normalized terms.

    Example:
      ["voltage", "напряжение", "voltage regulator"] ->
        (voltage OR напряжение OR "voltage regulator")
    """
    uniq: List[str] = []
    seen: Set[str] = set()
    for t in terms:
        nt = normalize_term(t)
        if not nt:
            continue
        if nt in seen:
            continue
        seen.add(nt)
        uniq.append(nt)

    if not uniq:
        return ""

    parts: List[str] = []
    for t in uniq:
        if _needs_fts_quotes(t):
            parts.append(f'"{_escape_fts_phrase(t)}"')
        else:
            parts.append(t)

    if len(parts) == 1:
        return parts[0]
    return "(" + " OR ".join(parts) + ")"


def build_simple_fts_query_from_groups(groups: Sequence[Sequence[str]]) -> str:
    """
    Convert OR-groups into an FTS5 simple-mode query joined with AND.

    Example:
      groups = [["thermal"], ["resistance", "rth"]] ->
        thermal AND (resistance OR rth)
    """
    clauses: List[str] = []
    for g in groups:
        clause = build_fts_or_group(list(g))
        if clause:
            clauses.append(clause)
    return " AND ".join(clauses).strip()


def tokenize_simple_query(normalized_query: str) -> List[str]:
    """
    Tokenize a normalized query in "simple mode" by whitespace.

    This mirrors the approach in step 1 query formation.
    """
    q = (normalized_query or "").strip()
    q = re.sub(r"\s+", " ", q).strip()
    return [t for t in q.split(" ") if t] if q else []
