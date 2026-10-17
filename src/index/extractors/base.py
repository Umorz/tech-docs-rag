"""
Base classes for file content extractors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    """
    One «page» of extracted text. For PDF this is a real page; for formats
    without natural pagination (TXT, MD, DOCX) the entire file is one page.
    """

    page_no: int  # 1-based, used in citations
    text: str
    is_ocr: bool = False  # True if text comes from OCR rather than the document's
                          # native text layer (currently only PDF supports this).


class Extractor:
    """Abstract extractor. Subclasses declare supported extensions."""

    extensions: tuple[str, ...] = ()

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        """Return list of ExtractedPage. Must be implemented by subclass."""
        raise NotImplementedError
