"""
Markdown (.md, .markdown) extractor.

Markdown is treated as plain text: full content of the file becomes a single
page. We deliberately do NOT strip Markdown markup, because for technical
searches the raw markup (code fences, link URLs, headings) can be semantically
relevant.
"""
from __future__ import annotations

from pathlib import Path

from .base import ExtractedPage, Extractor
from .txt import _read_text_any_encoding
from .. import text_clean


class MdExtractor(Extractor):
    extensions = (".md", ".markdown")

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        text = _read_text_any_encoding(file_path)
        text = text_clean.clean_page_text(text)
        return [ExtractedPage(page_no=1, text=text, is_ocr=False)]
