"""Общая подготовка: маленький корпус из PDF, DOCX и TXT и индекс по нему.

Индекс строится теми же скриптами, что и в реальной работе (шаги A, B, D),
но с хеш-эмбеддером: он детерминирован и не требует Ollama, поэтому тесты
идут где угодно, в том числе в CI.
"""
import subprocess
import sys
from pathlib import Path

import docx
import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]


def make_corpus(corpus: Path) -> None:
    (corpus / "sub").mkdir(parents=True)
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "LM317 adjustable voltage regulator", fontsize=12)
    page.insert_text((72, 100), "Operating temperature range: -40 C to +125 C", fontsize=12)
    page.insert_text((72, 128), "Output voltage adjustable from 1.25 V to 37 V", fontsize=12)
    pdf.save(corpus / "lm317.pdf")

    word = docx.Document()
    word.add_paragraph("NE555 timer: supply voltage 4.5 V to 16 V.")
    word.save(corpus / "sub" / "ne555.docx")

    (corpus / "notes.txt").write_text("Pull-up resistor 10 kOhm on the reset pin.\n", encoding="utf-8")


def run_step(script: str, *args: str) -> str:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                         capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def build_index(corpus: Path, db: Path) -> None:
    run_step("step_a_scan_corpus.py", "--corpus-root", str(corpus), "--db", str(db))
    run_step("step_b_index_pages.py", "--corpus-root", str(corpus), "--db", str(db))
    run_step("step_d_embed_chunks.py", "--db", str(db), "--embedder", "hash", "--min-chunk-chars", "10")


@pytest.fixture(scope="session")
def indexed(tmp_path_factory):
    """(папка корпуса, путь к базе) - индекс строится один раз на все тесты."""
    base = tmp_path_factory.mktemp("rag")
    corpus, db = base / "corpus", base / "index.sqlite"
    make_corpus(corpus)
    build_index(corpus, db)
    return corpus, db
