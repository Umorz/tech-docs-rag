"""Сквозная проверка без внешних сервисов: индексация своего корпуса -> поиск -> ссылка."""
import subprocess
import sys

from src.index.db import connect
from src.index.embeddings import HashEmbedder
from src.index.hybrid import HybridOptions, hybrid_search_rrf
from tests.conftest import ROOT, build_index, run_step


def test_all_formats_indexed(indexed):
    corpus, db = indexed
    with connect(db) as conn:
        paths = sorted(r[0].replace("\\", "/") for r in conn.execute("select rel_path from documents"))
    assert paths == ["lm317.pdf", "notes.txt", "sub/ne555.docx"]


def test_lexical_search_cites_page_and_line(indexed):
    _, db = indexed
    out = run_step("step_c_query.py", "--db", str(db), "operating temperature")
    assert "lm317.pdf:1:2" in out


def test_hybrid_search_fuses_both_channels(indexed):
    # Хеш-эмбеддинги не несут смысла, поэтому порядок семантического канала здесь
    # случаен и не проверяется: проверяем, что оба канала отработали и слились.
    _, db = indexed
    with connect(db) as conn:
        plan, hits = hybrid_search_rrf(conn, "operating temperature", embedder=HashEmbedder(),
                                       options=HybridOptions(final_k=10))
    assert plan.semantic_enabled and plan.semantic_error is None
    assert any(h.sem_rank is not None for h in hits)
    best_lex = next(h for h in hits if h.lex_rank == 1)
    assert (best_lex.key.rel_path, best_lex.key.page_no, best_lex.key.start_line_no) == ("lm317.pdf", 1, 2)


def test_new_file_is_picked_up_on_rescan(tmp_path):
    corpus, db = tmp_path / "corpus", tmp_path / "index.sqlite"
    corpus.mkdir()
    (corpus / "a.txt").write_text("first document about capacitors\n", encoding="utf-8")
    build_index(corpus, db)
    (corpus / "b.txt").write_text("second document about inductors\n", encoding="utf-8")
    build_index(corpus, db)
    out = run_step("step_c_query.py", "--db", str(db), "inductors")
    assert "b.txt" in out


def run_repl(db, commands: str):
    out = subprocess.run([sys.executable, "Main.py", "--db", str(db), "--mode", "hybrid", "--sem-backend", "hash",
                          "--llm", "off", "--no-startup-prompts"],
                         input=commands, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    return out.returncode, out.stdout + out.stderr


def test_repl_answers_query_with_citation(indexed):
    _, db = indexed
    code, out = run_repl(db, ":sem hash_dim 384\noperating temperature\n:quit\n")
    assert code == 0, out
    assert "lm317.pdf:1:2" in out


def test_repl_exits_cleanly_when_input_ends(tmp_path):
    # без индекса консоль спрашивает, строить ли его; конец ввода - это «нет», а не падение
    out = subprocess.run([sys.executable, "Main.py", "--db", str(tmp_path / "none.sqlite")],
                         input="", capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Traceback" not in out.stderr
