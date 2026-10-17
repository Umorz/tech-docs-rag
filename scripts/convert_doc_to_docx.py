"""
Batch-convert legacy .doc files in corpus/ to .docx via LibreOffice.

Requires LibreOffice installed:
  https://www.libreoffice.org/download/

The script autodetects soffice.exe (PATH or default install locations).
Originals (.doc) are NOT deleted — you can clean them up manually after
verifying that the corresponding .docx files indexed correctly.

Usage:
  python scripts/convert_doc_to_docx.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _PROJECT_ROOT / "corpus"


def find_soffice() -> Path | None:
    """Locate soffice.exe on Windows / soffice on Linux/Mac."""
    for name in ("soffice", "soffice.exe"):
        p = shutil.which(name)
        if p:
            return Path(p)
    candidates = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        Path("/usr/bin/soffice"),
        Path("/usr/local/bin/soffice"),
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def convert_one(soffice: Path, doc_path: Path, out_dir: Path) -> bool:
    """Run soffice headless conversion. Returns True if .docx was produced."""
    try:
        proc = subprocess.run(
            [
                str(soffice),
                "--headless",
                "--convert-to", "docx",
                "--outdir", str(out_dir),
                str(doc_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print(f"  [error] timeout after 180s", flush=True)
        return False
    except Exception as e:
        print(f"  [error] {type(e).__name__}: {e}", flush=True)
        return False

    out_path = out_dir / (doc_path.stem + ".docx")
    if proc.returncode == 0 and out_path.exists():
        return True

    err = (proc.stderr or proc.stdout or "").strip()
    if err:
        print(f"  [error] soffice: {err[:200]}", flush=True)
    return False


def main() -> int:
    if not _CORPUS.exists():
        print(f"[error] Corpus folder not found: {_CORPUS}", file=sys.stderr)
        return 1

    docs = sorted(p for p in _CORPUS.rglob("*.doc") if p.suffix.lower() == ".doc")
    if not docs:
        # Тихо выходим — нет .doc, нечего делать.
        return 0

    soffice = find_soffice()
    if soffice is None:
        print(
            f"[warn] Найдено {len(docs)} файлов .doc, но LibreOffice не установлен.\n"
            "       Эти файлы не будут проиндексированы.\n"
            "       Установи LibreOffice: https://www.libreoffice.org/download/",
            file=sys.stderr,
        )
        return 2

    print(f"[doc->docx] LibreOffice: {soffice}", flush=True)
    print(f"[doc->docx] Найдено .doc файлов: {len(docs)}", flush=True)

    converted = 0
    skipped = 0
    failed = 0

    for i, doc in enumerate(docs, 1):
        out_path = doc.with_suffix(".docx")
        if out_path.exists():
            print(f"[{i}/{len(docs)}] SKIP (уже есть .docx): {doc.name}", flush=True)
            skipped += 1
            continue

        print(f"[{i}/{len(docs)}] convert: {doc.name}", flush=True)
        if convert_one(soffice, doc, doc.parent):
            converted += 1
        else:
            failed += 1

    print(
        f"[doc->docx] Готово. Сконвертировано: {converted}, "
        f"пропущено: {skipped}, ошибки: {failed}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
