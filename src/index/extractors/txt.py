"""
Plain text (.txt) extractor with multi-encoding fallback.
"""
from __future__ import annotations

from pathlib import Path

from .. import text_clean
from .base import ExtractedPage, Extractor


# Common encodings, tried in order until decoding succeeds without errors.
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251", "cp866", "latin-1")


def _read_text_any_encoding(path: Path) -> str:
    raw_bytes = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    # Final fallback: replace undecodable bytes.
    return raw_bytes.decode("utf-8", errors="replace")


class TxtExtractor(Extractor):
    extensions = (".txt",)

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        text = _read_text_any_encoding(file_path)
        text = text_clean.clean_page_text(text)
        return [ExtractedPage(page_no=1, text=text, is_ocr=False)]
