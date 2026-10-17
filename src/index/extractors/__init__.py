"""
Extractor registry: dispatch text extraction by file extension.

Each extractor returns a list of ExtractedPage objects (1-based page numbering).
For formats without natural pages (TXT, MD, DOCX), the whole file is returned
as a single page.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Extractor, ExtractedPage
from .docx_extractor import DocxExtractor
from .md import MdExtractor
from .pdf import PdfExtractor
from .txt import TxtExtractor


# Registry of available extractors keyed by lower-case file extension.
_REGISTRY: dict[str, Extractor] = {}
for _ext_cls in (PdfExtractor, TxtExtractor, MdExtractor, DocxExtractor):
    _inst = _ext_cls()
    for _ext in _inst.extensions:
        _REGISTRY[_ext.lower()] = _inst


def get_extractor(file_path: Path) -> Optional[Extractor]:
    """
    Return an extractor for the given file path, or None if no extractor is
    registered for its extension.
    """
    return _REGISTRY.get(file_path.suffix.lower())


def supported_extensions() -> tuple[str, ...]:
    """Tuple of all supported file extensions (lower-case, with dot)."""
    return tuple(sorted(_REGISTRY.keys()))


__all__ = [
    "ExtractedPage",
    "Extractor",
    "get_extractor",
    "supported_extensions",
]
