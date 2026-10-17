from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


# Common whitespace characters found in PDFs/OCR that we want to normalize.
_NBSP = "\u00A0"
_NARROW_NBSP = "\u202F"
_THIN_SPACE = "\u2009"
_ZERO_WIDTH_SPACE = "\u200B"
_SOFT_HYPHEN = "\u00AD"

# Micro sign and Greek mu frequently appear in datasheets and OCR output.
_MICRO_SIGN = "\u00B5"  # µ
_GREEK_MU = "\u03BC"  # μ


def normalize_unicode(text: str) -> str:
    """
    Normalize text to a more search-friendly form.

    - Applies NFKC normalization (e.g. compatibility forms).
    - Normalizes odd whitespace characters.
    - Removes zero-width characters and soft hyphens.

    This function intentionally does NOT lower-case the text; keep case as-is for
    component part numbers, signal names, etc. Query-time can apply its own rules.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Remove/normalize invisible and special spacing characters.
    text = text.replace("\x00", "")  # nulls can appear in extracted PDF text
    text = text.replace(_ZERO_WIDTH_SPACE, "")
    text = text.replace(_SOFT_HYPHEN, "")  # soft hyphen should not affect search

    # Normalize various spaces to regular space.
    text = (
        text.replace(_NBSP, " ")
        .replace(_NARROW_NBSP, " ")
        .replace(_THIN_SPACE, " ")
    )

    # Normalize micro symbol variants to 'u' for better stability in search queries.
    # Example: 10µF / 10μF -> 10uF
    text = text.replace(_MICRO_SIGN, "u").replace(_GREEK_MU, "u")

    return text


def normalize_newlines(text: str) -> str:
    """Convert Windows/Mac newlines to '\\n'."""
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-zА-Яа-я0-9])-\n([A-Za-zА-Яа-я0-9])")


def fix_hyphenation(text: str) -> str:
    """
    Merge words split by hyphen + newline:
      "confi-\\nguration" -> "configuration"

    This is a heuristic; it helps datasheets but may occasionally merge things it
    shouldn't. For MVP it's usually worth it.
    """
    if not text:
        return ""
    # Repeat until stable (handles multiple wrapped hyphenations).
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    return text


_TRAILING_SPACES_RE = re.compile(r"[ \t]+\n")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_page_text(text: str) -> str:
    """
    Clean text extracted from a PDF page (non-OCR path).

    Steps:
    - Unicode normalization
    - newline normalization
    - hyphenation fix for wrapped words
    - trims trailing spaces on lines
    - collapses excessive blank lines (keeps paragraph structure)
    """
    text = normalize_unicode(text)
    text = normalize_newlines(text)
    text = fix_hyphenation(text)

    # Remove trailing spaces before newline to stabilize storage and search.
    text = _TRAILING_SPACES_RE.sub("\n", text)

    # Avoid huge blank areas due to layout; keep at most 2 consecutive newlines.
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)

    return text.strip()


def clean_ocr_text(text: str) -> str:
    """
    Clean text produced by OCR.

    OCR output usually has more noise; for now we apply the same cleaning as for
    page text. This function exists separately so we can tune OCR-specific rules later.
    """
    return clean_page_text(text)


def normalize_text(text: str) -> str:
    """
    Stable normalization API used by indexing/retrieval code.

    This is the function other modules should call to normalize page text before:
      - storing it in SQLite
      - splitting it into quote-friendly lines
      - feeding it to full-text search

    Today it is equivalent to `clean_page_text()`. Keeping a dedicated name makes it
    easy to evolve the rules without changing call sites.
    """
    return clean_page_text(text)


_INLINE_WS_RE = re.compile(r"[ \t]+")


def split_lines(text: str, min_len: int = 2) -> List[str]:
    """
    Split normalized text into lines suitable for the `lines` table.

    - Normalizes the input with `normalize_text()`
    - Trims each line
    - Collapses internal whitespace (spaces/tabs) within a line
    - Drops empty/too-short lines
    """
    norm = normalize_text(text)
    if not norm:
        return []

    out: List[str] = []
    for raw in norm.split("\n"):
        line = raw.strip()
        if not line:
            continue

        line = _INLINE_WS_RE.sub(" ", line).strip()
        if len(line) < int(min_len):
            continue

        out.append(line)

    return out


def iter_line_windows(lines: Iterable[str], window: int = 2) -> Iterable[str]:
    """
    Yield sliding windows of `window` consecutive lines joined by '\\n'.

    This is useful later (Query-time) to return a short quote context around a match.
    """
    if window <= 0:
        raise ValueError("window must be >= 1")

    buf: List[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) == window:
            yield "\n".join(buf)
            buf.pop(0)


_MULTI_SPACES_RE = re.compile(r"[ \t]{2,}")


def clean_line(line: str) -> str:
    """
    Clean a single line for `lines` table storage:
    - normalize unicode
    - collapse internal whitespace
    - strip
    """
    line = normalize_unicode(line)
    line = line.replace("\n", " ")
    line = _MULTI_SPACES_RE.sub(" ", line)
    return line.strip()


def split_lines(text: str, *, min_len: int = 2) -> List[str]:
    """
    Split page text into a list of cleaned lines suitable for precise quoting.

    Args:
        text: cleaned page text (recommended: output of clean_page_text/clean_ocr_text).
        min_len: minimum length (in characters) of a line to keep.

    Returns:
        List of cleaned, non-empty lines.
    """
    text = normalize_newlines(text)
    raw_lines = text.split("\n")
    out: List[str] = []
    for raw in raw_lines:
        line = clean_line(raw)
        if len(line) < min_len:
            continue
        out.append(line)
    return out


def text_len_for_threshold(text: str) -> int:
    """
    Length measure used to decide if a page is "empty" and needs OCR.

    We count non-whitespace characters because extracted text may contain many spaces/newlines.
    """
    if not text:
        return 0
    return len("".join(ch for ch in text if not ch.isspace()))


def is_effectively_empty(text: str, *, threshold: int) -> bool:
    """
    Decide whether extracted page text is too short to be useful.

    Use this for the "if text_len < threshold -> OCR" rule in Step B.
    """
    return text_len_for_threshold(text) < int(threshold)


def join_lines_for_preview(lines: Iterable[str], *, max_lines: int = 6) -> str:
    """
    Utility for debugging/logging: join first N lines into a small preview.
    Not used by the index itself, but handy during development.
    """
    buf: List[str] = []
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        buf.append(line)
    return "\n".join(buf)