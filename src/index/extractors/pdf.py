"""
PDF extractor (PyMuPDF).

Note: this extractor returns text only; OCR fallback for empty pages stays in
`scripts/step_b_index_pages.py` because it requires extra parameters (DPI,
language, tesseract config) that are CLI-controlled.
"""
from __future__ import annotations

from pathlib import Path

from .. import text_clean
from .base import ExtractedPage, Extractor


class PdfExtractor(Extractor):
    extensions = (".pdf",)

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        try:
            import fitz  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF extraction. "
                "Install with: pip install pymupdf. "
                f"Original error: {e}"
            ) from e

        pages: list[ExtractedPage] = []
        doc = fitz.open(str(file_path))
        try:
            for i in range(int(doc.page_count)):
                page = doc.load_page(i)
                raw = page.get_text("text") or ""
                cleaned = text_clean.clean_page_text(raw)
                pages.append(ExtractedPage(page_no=i + 1, text=cleaned, is_ocr=False))
        finally:
            doc.close()
        return pages
