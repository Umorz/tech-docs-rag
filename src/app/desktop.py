"""
Desktop GUI for the RAG search system (no browser, native window).

Запуск:
    pip install customtkinter
    python -m src.app.desktop

Использует те же модули, что REPL и Streamlit-версия:
src.index.{db, embeddings, hybrid, search, llm}.
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import customtkinter as ctk

from src.index.db import connect
from src.index.embeddings import create_embedder
from src.index.hybrid import HybridOptions, hybrid_search_rrf
from src.index.llm import (
    LLMContextItem,
    LLMError,
    OllamaChat,
    OllamaChatConfig,
    build_rag_messages,
)
from src.index.search import SearchOptions

DB_PATH = _PROJECT_ROOT / "data" / "index.sqlite"

# Стабильный label embedder'а (см. embeddings.stable_model_label_for_embedder).
# Используется для подсчёта embeddings по каждому документу в сайдбаре.
EMBED_MODEL_LABEL = "ollama:mxbai-embed-large"


# -----------------------------
# Helpers (без декораторов Streamlit)
# -----------------------------


def _has_cyrillic(text: str, min_chars: int = 2) -> bool:
    n = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return n >= min_chars


def _list_indexed_docs() -> list[tuple[str, int, int, int, str]]:
    """
    Возвращает список документов с полной информацией для сайдбара:
      (rel_path, content_pages, chunks, embeddings_for_current_model, status)

    content_pages — реальные страницы (исключая синтетическую title-страницу
    с page_no=0).
    embeddings — сколько чанков документа уже embedded под текущей моделью.
    """
    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT
              d.rel_path,
              (SELECT COUNT(*) FROM pages p
                 WHERE p.doc_id = d.doc_id AND p.page_no > 0) AS content_pages,
              (SELECT COUNT(*) FROM chunks c
                 JOIN pages p ON p.page_id = c.page_id
                 WHERE p.doc_id = d.doc_id) AS chunks_total,
              (SELECT COUNT(*) FROM chunk_embeddings e
                 JOIN chunks c ON c.chunk_id = e.chunk_id
                 JOIN pages p ON p.page_id = c.page_id
                 WHERE p.doc_id = d.doc_id AND e.model = ?) AS emb_count,
              d.status
            FROM documents d
            ORDER BY d.rel_path
            """,
            (EMBED_MODEL_LABEL,),
        ).fetchall()
    finally:
        conn.close()
    return [
        (
            str(r[0]),
            int(r[1] or 0),
            int(r[2] or 0),
            int(r[3] or 0),
            str(r[4] or ""),
        )
        for r in rows
    ]


def _get_chat_models() -> list[str]:
    """Список доступных в Ollama моделей чата (без эмбеддеров)."""
    fallback = ["t-pro-it"]
    try:
        probe = OllamaChat(OllamaChatConfig(model="probe"))
        if not probe.ping():
            return fallback
        all_models = probe.list_models()
    except Exception:
        return fallback
    emb_kw = ("embed", "nomic", "mxbai", "bge")
    chat = [m for m in all_models if not any(k in m.lower() for k in emb_kw)]
    chat.sort(
        key=lambda n: (0 if "t-pro" in n.lower() else 1 if "t-lite" in n.lower() else 2, n.lower())
    )
    return chat or fallback


def _translate_query(query: str, model: str) -> str | None:
    if not _has_cyrillic(query):
        return None
    try:
        client = OllamaChat(OllamaChatConfig(model=model, temperature=0.2))
        if not client.ping():
            return None
        sys_prompt = (
            "You translate Russian search queries into concise English search queries. "
            "Output ONLY the English query, on a single line, without quotes or "
            "explanations. Keep names, identifiers, dates, numbers and units unchanged. "
            "Use natural English suitable for general document search."
        )
        raw = client.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query.strip()},
            ]
        )
    except Exception:
        return None
    out = (raw or "").strip().split("\n", 1)[0].strip()
    for q in ('"', "'", "«", "»", "`"):
        out = out.strip(q)
    out = out.strip()
    if not out or len(out) > 500 or _has_cyrillic(out, 4):
        return None
    return out


_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = create_embedder(
            preferred=["ollama"], ollama_model="mxbai-embed-large", normalized=True
        )
    return _EMBEDDER


def _retrieve_hybrid(query: str, limit: int) -> list[dict]:
    conn = connect(DB_PATH)
    try:
        _, fused = hybrid_search_rrf(
            conn,
            query,
            lexical_options=SearchOptions(target="lines"),
            embedder=_get_embedder(),
            options=HybridOptions(final_k=limit),
        )
        return [
            {
                "rel_path": h.key.rel_path,
                "page_no": int(h.key.page_no),
                "start_line": int(h.key.start_line_no),
                "end_line": int(h.key.end_line_no),
                "text": h.text or "",
                "score": float(h.score),
            }
            for h in fused
        ]
    finally:
        conn.close()


def _expand(results: list[dict], top_n: int, n_lines: int = 15) -> list[dict]:
    out = []
    conn = connect(DB_PATH)
    try:
        for r in results[:top_n]:
            es = max(1, int(r["start_line"]) - n_lines)
            ee = int(r["end_line"]) + n_lines
            rows = conn.execute(
                """
                SELECT l.line_no, l.text FROM lines l
                JOIN pages p ON p.page_id = l.page_id
                JOIN documents d ON d.doc_id = p.doc_id
                WHERE d.rel_path = ? AND p.page_no = ?
                  AND l.line_no BETWEEN ? AND ?
                ORDER BY l.line_no
                """,
                (r["rel_path"], r["page_no"], es, ee),
            ).fetchall()
            if rows:
                text = "\n".join(str(t) for _, t in rows)
                ls, le = int(rows[0][0]), int(rows[-1][0])
            else:
                text, ls, le = r["text"], r["start_line"], r["end_line"]
            out.append({**r, "text_expanded": text, "start_exp": ls, "end_exp": le})
    finally:
        conn.close()
    return out


# -----------------------------
# Main application window
# -----------------------------


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class RAGSearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Универсальный поиск")
        self.geometry("1280x820")
        self.minsize(960, 640)

        self.stream_queue: queue.Queue = queue.Queue()
        self._search_running = False
        self._watcher = None
        self._reindex_lock = threading.Lock()
        self._reindex_dirty = False
        self._reindex_dirty_lock = threading.Lock()

        self._build_ui()
        self.after(100, self._poll_stream_queue)
        # Запуск фонового наблюдения за корпусом после построения UI.
        self.after(500, self._start_corpus_watcher)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI build ----------

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        # Main area
        self.main = ctk.CTkFrame(self, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self._build_main()

    def _build_sidebar(self):
        title = ctk.CTkLabel(
            self.sidebar,
            text="📁 Коллекция файлов",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        title.pack(padx=15, pady=(15, 4), anchor="w")

        # Статус наблюдения (по умолчанию — пусто, появляется при индексации)
        self.watch_status = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="#888",
        )
        self.watch_status.pack(padx=15, pady=(0, 4), anchor="w")

        self.docs_list = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.docs_list.pack(fill="both", expand=True, padx=10, pady=5)

        self.docs_count_label = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#888",
        )
        self.docs_count_label.pack(padx=15, pady=(5, 4), anchor="w")

        # Кнопка обычного reindex — добивает пропущенные embeddings
        # и забирает новые/изменённые файлы. Не разрушительно.
        self.reindex_btn = ctk.CTkButton(
            self.sidebar,
            text="↻ Обновить индекс",
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            command=self._on_manual_reindex,
        )
        self.reindex_btn.pack(padx=15, pady=(0, 4), fill="x")

        # Кнопка полной пересборки — для случаев когда что-то пошло вконец не так.
        self.force_btn = ctk.CTkButton(
            self.sidebar,
            text="⟳ Полная пересборка (медленно)",
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color="transparent",
            border_width=1,
            text_color=("#888", "#aaa"),
            command=self._on_force_rebuild,
        )
        self.force_btn.pack(padx=15, pady=(0, 12), fill="x")

        self._refresh_sidebar()

    def _refresh_sidebar(self):
        """Clear docs_list and re-populate from DB (called on changes)."""
        for w in self.docs_list.winfo_children():
            w.destroy()

        docs = _list_indexed_docs()

        searchable = 0  # документов, реально доступных для семантического поиска
        missing_emb = 0
        errors = 0

        for rel_path, pages_n, chunks_n, emb_n, status in docs:
            # Определяем «здоровье» документа
            if status == "error":
                icon = "⚠️"
                tail_color = "#d93025"
                tail = f"   статус: error"
                errors += 1
            elif chunks_n == 0:
                icon = "⏳"
                tail_color = "#b06000"
                tail = f"   {pages_n} стр., нет чанков (нужна реиндексация)"
                missing_emb += 1
            elif emb_n == 0:
                icon = "⚠️"
                tail_color = "#b06000"
                tail = (
                    f"   {pages_n} стр., {chunks_n} чанков, "
                    f"embeddings: 0 — НЕ в поиске"
                )
                missing_emb += 1
            elif emb_n < chunks_n:
                icon = "🟡"
                tail_color = "#b06000"
                tail = (
                    f"   {pages_n} стр., embeddings {emb_n}/{chunks_n} — частично"
                )
                searchable += 1
            else:
                icon = "📄"
                tail_color = "#888"
                tail = f"   {pages_n} стр., {emb_n} embeddings ✓"
                searchable += 1

            frame = ctk.CTkFrame(self.docs_list, fg_color="transparent")
            frame.pack(fill="x", pady=2)
            ctk.CTkLabel(
                frame,
                text=f"{icon} {rel_path}",
                font=ctk.CTkFont(size=12),
                anchor="w",
                justify="left",
                wraplength=260,
            ).pack(fill="x", anchor="w")
            ctk.CTkLabel(
                frame,
                text=tail,
                font=ctk.CTkFont(size=10),
                text_color=tail_color,
                anchor="w",
            ).pack(fill="x", anchor="w")

        total = len(docs)
        summary = f"В поиске: {searchable}/{total}"
        if missing_emb:
            summary += f"  ·  без embeddings: {missing_emb}"
        if errors:
            summary += f"  ·  ошибок: {errors}"
        self.docs_count_label.configure(text=summary)

    def _build_main(self):
        # Title
        title = ctk.CTkLabel(
            self.main,
            text="🔎 Универсальный поиск",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        # Search row
        search_row = ctk.CTkFrame(self.main, fg_color="transparent")
        search_row.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="ew")
        search_row.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(
            search_row,
            placeholder_text="Введите запрос на любом языке",
            font=ctk.CTkFont(size=14),
            height=40,
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self._on_search())

        self.search_btn = ctk.CTkButton(
            search_row,
            text="Найти",
            width=100,
            height=40,
            font=ctk.CTkFont(size=14),
            command=self._on_search,
        )
        self.search_btn.grid(row=0, column=1)

        # Controls row
        controls_row = ctk.CTkFrame(self.main, fg_color="transparent")
        controls_row.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="w")

        ctk.CTkLabel(controls_row, text="Результатов:", font=ctk.CTkFont(size=12)).pack(
            side="left", padx=(0, 5)
        )
        self.limit_var = ctk.StringVar(value="10")
        self.limit_menu = ctk.CTkOptionMenu(
            controls_row,
            values=["3", "5", "10", "15", "20"],
            variable=self.limit_var,
            width=70,
        )
        self.limit_menu.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(controls_row, text="Модель:", font=ctk.CTkFont(size=12)).pack(
            side="left", padx=(0, 5)
        )
        models = _get_chat_models()
        self.model_var = ctk.StringVar(value=models[0])
        self.model_menu = ctk.CTkOptionMenu(
            controls_row, values=models, variable=self.model_var, width=200
        )
        self.model_menu.pack(side="left", padx=(0, 20))

        # Переключатель "Ответ от ИИ" — позволяет отключить генерацию ответа
        # и видеть только что нашли embeddings (без ожидания LLM).
        self.ai_enabled_var = ctk.BooleanVar(value=True)
        self.ai_switch = ctk.CTkSwitch(
            controls_row,
            text="Ответ от ИИ",
            variable=self.ai_enabled_var,
            font=ctk.CTkFont(size=12),
        )
        self.ai_switch.pack(side="left")

        # Translated query line (initially hidden)
        self.translated_label = ctk.CTkLabel(
            self.main,
            text="",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="#888",
            anchor="w",
        )
        self.translated_label.grid(row=3, column=0, padx=20, sticky="ew")
        self.translated_label.grid_remove()

        # Status line
        self.status_label = ctk.CTkLabel(
            self.main, text="", font=ctk.CTkFont(size=11), text_color="#888", anchor="w"
        )

        # Results frame (scrollable)
        self.results_frame = ctk.CTkScrollableFrame(self.main, label_text="")
        self.results_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.main.grid_rowconfigure(4, weight=1)

    # ---------- Search flow ----------

    def _on_search(self):
        if self._search_running:
            return
        query = self.search_entry.get().strip()
        if not query:
            return

        # Reset UI
        self._clear_results()
        self.translated_label.grid_remove()
        self._search_running = True
        self.search_btn.configure(state="disabled", text="…")

        # Run in background thread
        threading.Thread(target=self._do_search, args=(query,), daemon=True).start()

    def _do_search(self, query: str):
        try:
            limit = int(self.limit_var.get())
            model = self.model_var.get()

            # Translate
            effective = query
            if _has_cyrillic(query):
                translated = _translate_query(query, model)
                if translated:
                    effective = translated
                    self.after(0, self._show_translated, translated)

            # Retrieve
            results = _retrieve_hybrid(effective, limit)
            ai_enabled = bool(self.ai_enabled_var.get())
            self.after(0, self._show_results, results, ai_enabled)

            # Stream LLM in another thread (только если включён переключатель)
            if results and ai_enabled:
                threading.Thread(
                    target=self._stream_llm,
                    args=(query, results, model),
                    daemon=True,
                ).start()
            else:
                self.after(0, self._reset_search_btn)
        except Exception as e:
            self.after(0, self._show_error, str(e))
            self.after(0, self._reset_search_btn)

    def _stream_llm(self, query: str, results: list[dict], model: str):
        try:
            expanded = _expand(results, top_n=6, n_lines=15)
            client = OllamaChat(OllamaChatConfig(model=model, temperature=0.2))
            if not client.ping():
                self.stream_queue.put(("error", "Сервер Ollama недоступен."))
                self.stream_queue.put(("done", None))
                return

            def _cite(r):
                if (
                    int(r["page_no"]) == 0
                    and int(r["start_exp"]) == 0
                    and int(r["end_exp"]) == 0
                ):
                    return f"{r['rel_path']} (название файла)"
                return (
                    f"{r['rel_path']}, стр. {r['page_no']}, "
                    f"строки {r['start_exp']}–{r['end_exp']}"
                )

            items = [
                LLMContextItem(
                    label=str(i + 1),
                    citation=_cite(r),
                    text=r["text_expanded"],
                )
                for i, r in enumerate(expanded)
            ]
            messages = build_rag_messages(query, items, max_chars_per_item=4000)

            self.stream_queue.put(("start", None))
            for delta in client.chat_stream(messages):
                self.stream_queue.put(("delta", delta))
            self.stream_queue.put(("done", None))
        except LLMError as e:
            self.stream_queue.put(("error", str(e)))
            self.stream_queue.put(("done", None))
        except Exception as e:
            self.stream_queue.put(("error", f"{type(e).__name__}: {e}"))
            self.stream_queue.put(("done", None))

    # ---------- Stream queue polling ----------

    def _poll_stream_queue(self):
        try:
            while True:
                kind, data = self.stream_queue.get_nowait()
                if kind == "start":
                    self._init_answer_block()
                elif kind == "delta":
                    self._append_answer(data)
                elif kind == "error":
                    self._show_error(data)
                elif kind == "done":
                    self._reset_search_btn()
        except queue.Empty:
            pass
        self.after(100, self._poll_stream_queue)

    # ---------- UI updates ----------

    def _clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.answer_textbox = None

    def _reset_search_btn(self):
        self._search_running = False
        self.search_btn.configure(state="normal", text="Найти")

    def _show_translated(self, text: str):
        self.translated_label.configure(text=f"Запрос для поиска: {text}")
        self.translated_label.grid()

    def _show_results(self, results: list[dict], ai_enabled: bool = True):
        if not results:
            ctk.CTkLabel(
                self.results_frame,
                text="По запросу ничего не найдено.",
                text_color="#888",
            ).pack(pady=10)
            return

        # 1. AI answer block FIRST (over results) — только если включён ИИ
        if ai_enabled:
            ai_header = ctk.CTkLabel(
                self.results_frame,
                text="⚡ Ответ от ИИ",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=("#1a73e8", "#8ab4f8"),
                anchor="w",
            )
            ai_header.pack(fill="x", pady=(0, 4))

            self.answer_textbox = ctk.CTkTextbox(
                self.results_frame, height=180, font=ctk.CTkFont(size=12), wrap="word"
            )
            self.answer_textbox.pack(fill="x", pady=(0, 4))
            self.answer_textbox.configure(state="normal")
            self.answer_textbox.insert("1.0", "Генерация ответа…")
            self.answer_textbox.configure(state="disabled")
            self._answer_text = ""
            self._answer_started = False

            # 2. Separator between AI and results
            sep = ctk.CTkFrame(
                self.results_frame, height=2, fg_color=("#d0d7de", "#444")
            )
            sep.pack(fill="x", pady=(10, 10))
        else:
            self.answer_textbox = None

        # 3. Results header + items
        header = ctk.CTkLabel(
            self.results_frame,
            text=f"📑 Результаты поиска  ({len(results)} фрагментов)",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        header.pack(fill="x", pady=(0, 8))

        for i, r in enumerate(results, 1):
            block = ctk.CTkFrame(self.results_frame, fg_color="transparent")
            block.pack(fill="x", pady=4)

            is_title_hit = (
                int(r["page_no"]) == 0
                and int(r["start_line"]) == 0
                and int(r["end_line"]) == 0
            )
            prefix = "🏷️" if is_title_hit else "📄"

            title = ctk.CTkLabel(
                block,
                text=f"[{i}] {prefix} {r['rel_path']}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=("#1a0dab", "#8ab4f8"),
                anchor="w",
            )
            title.pack(fill="x", anchor="w")

            if is_title_hit:
                cite_text = (
                    f"совпадение с названием файла  ·  релевантность {r['score']:.4f}"
                )
            else:
                cite_text = (
                    f"страница {r['page_no']}, строки {r['start_line']}–{r['end_line']}"
                    f"  ·  релевантность {r['score']:.4f}"
                )
            citation = ctk.CTkLabel(
                block,
                text=cite_text,
                font=ctk.CTkFont(size=10),
                text_color=("#006621", "#81c995"),
                anchor="w",
            )
            citation.pack(fill="x", anchor="w")

            snippet = r["text"][:500] + ("…" if len(r["text"]) > 500 else "")
            text_label = ctk.CTkLabel(
                block,
                text=snippet,
                font=ctk.CTkFont(size=11),
                anchor="w",
                justify="left",
                wraplength=750,
            )
            text_label.pack(fill="x", anchor="w", pady=(2, 4))

    def _init_answer_block(self):
        # AI-блок теперь создаётся в _show_results заранее (над результатами).
        # Этот метод оставлен для совместимости со событием "start" из стримера.
        pass

    def _append_answer(self, delta: str):
        if not hasattr(self, "answer_textbox") or self.answer_textbox is None:
            return
        if not getattr(self, "_answer_started", False):
            # Очищаем заглушку "Генерация ответа…"
            self.answer_textbox.configure(state="normal")
            self.answer_textbox.delete("1.0", "end")
            self.answer_textbox.configure(state="disabled")
            self._answer_started = True

        self._answer_text += delta
        self.answer_textbox.configure(state="normal")
        self.answer_textbox.delete("1.0", "end")
        self.answer_textbox.insert("1.0", self._answer_text)
        self.answer_textbox.configure(state="disabled")
        self.answer_textbox.see("end")

    # ---------- Corpus watcher (background) ----------

    def _run_reindex_serialized(self):
        """
        Запустить переиндексацию в фоне с защитой от параллельных запусков.
        Если индексация уже идёт — лишь поднимает флаг dirty: текущий
        исполнитель после окончания заметит флаг и сделает ещё один проход.
        Это нужно, потому что пока step_b/step_d работают, новые файлы
        могут прилетать в corpus, и их нужно проиндексировать.
        """
        from src.index.watcher import run_reindex

        with self._reindex_dirty_lock:
            self._reindex_dirty = True

        if not self._reindex_lock.acquire(blocking=False):
            print("[gui] Reindex already running — queued another pass.", flush=True)
            return

        def worker():
            try:
                while True:
                    with self._reindex_dirty_lock:
                        if not self._reindex_dirty:
                            break
                        self._reindex_dirty = False

                    self.after(0, lambda: self.watch_status.configure(
                        text="↻ Идёт индексация (см. cmd-окно)..."
                    ))
                    try:
                        run_reindex(_PROJECT_ROOT, enable_ocr=True, ocr_lang="rus+eng")
                    except Exception as e:
                        print(f"[gui] Reindex error: {e}", flush=True)
                    self.after(0, self._refresh_sidebar)

                self.after(0, lambda: self.watch_status.configure(text=""))
            finally:
                self._reindex_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def _force_mark_all_changed(self) -> int:
        """
        Принудительно помечает ВСЕ документы как 'changed', чтобы step_b
        перечитал их с нуля. Используется кнопкой ручной переиндексации.
        Returns: количество затронутых документов.
        """
        try:
            conn = connect(DB_PATH)
            try:
                cur = conn.execute(
                    "UPDATE documents SET status='changed' "
                    "WHERE status IN ('indexed','unchanged','error','new','changed')"
                )
                n = int(cur.rowcount or 0)
                conn.commit()
                return n
            finally:
                conn.close()
        except Exception as e:
            print(f"[gui] Failed to mark documents as changed: {e}", flush=True)
            return 0

    def _start_corpus_watcher(self):
        """
        Запускает наблюдение за corpus/ в фоне. При появлении/изменении файла
        автоматически вызывает Шаг A + B + D и обновляет боковую панель.

        На старте также проверяет, не остались ли документы со статусом
        new/changed (например, если в прошлый раз приложение закрыли посреди
        индексации) и автоматически завершает их обработку.
        """
        try:
            from src.index.watcher import start_watching
        except ImportError:
            return  # watchdog не установлен — фича недоступна, приложение работает

        corpus = _PROJECT_ROOT / "corpus"
        if not corpus.exists():
            return

        try:
            self._watcher = start_watching(
                corpus, self._run_reindex_serialized, debounce_seconds=3.0
            )
            self.watch_status.configure(text="● Авто-индексация активна")
            self.after(4000, lambda: self.watch_status.configure(text=""))
        except Exception:
            pass

        # Если осталась незавершённая работа с прошлого сеанса — продолжить.
        self.after(1500, self._continue_unfinished_indexing)

    def _continue_unfinished_indexing(self):
        """Если есть документы со статусом new/changed, запустить индексацию."""
        try:
            conn = connect(DB_PATH)
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE status IN ('new','changed')"
                ).fetchone()
                unfinished = int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception:
            return

        if unfinished > 0:
            self._run_reindex_serialized()

    def _on_manual_reindex(self):
        """
        Запуск нормального пайплайна step_a + step_b + step_d.
        Step_d в incremental-режиме сам добьёт пропущенные embeddings.
        """
        print("\n[gui] Manual reindex triggered.", flush=True)
        self._run_reindex_serialized()

    def _on_force_rebuild(self):
        """
        ПОЛНАЯ пересборка: помечаем все документы как 'changed' и запускаем
        пайплайн. Step_b удалит старые страницы и перечитает файлы с нуля.
        Деструктивно — старые embeddings слетят и будут пересобраны.
        """
        n = self._force_mark_all_changed()
        print(
            f"\n[gui] Force rebuild: {n} documents marked as 'changed'.",
            flush=True,
        )
        self._run_reindex_serialized()

    def _on_close(self):
        """Aккуратно останавливаем watcher при закрытии окна."""
        # 1) Останавливаем наблюдение, чтобы не плодить новые задачи.
        if self._watcher is not None:
            try:
                self._watcher.stop()
                self._watcher.join(timeout=2.0)
            except Exception:
                pass

        # 2) Если сейчас идёт reindex — даём ему секунду закончить запись в БД,
        #    чтобы не оборвать SQLite-транзакцию посередине.
        if self._reindex_lock.locked():
            try:
                # Самим лочиться не надо — просто ждём короткий таймаут.
                import time as _t
                deadline = _t.time() + 1.5
                while self._reindex_lock.locked() and _t.time() < deadline:
                    _t.sleep(0.1)
            except Exception:
                pass

        self.destroy()

    def _show_error(self, message: str):
        err = ctk.CTkLabel(
            self.results_frame,
            text=f"⚠ {message}",
            text_color=("#d93025", "#f28b82"),
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        err.pack(fill="x", pady=5)


def _preflight_checks() -> None:
    """Печатает в cmd-окно состояние ключевых зависимостей и Ollama."""
    print("=" * 60, flush=True)
    print("Универсальный поиск — desktop GUI", flush=True)
    print("Это окно показывает прогресс индексации.", flush=True)
    print("Не закрывай его, пока работает приложение.", flush=True)
    print("-" * 60, flush=True)

    # 1) python-docx (для .docx)
    try:
        import docx  # noqa: F401
        print("[ok]   python-docx установлен — .docx будут индексироваться", flush=True)
    except ImportError:
        print(
            "[WARN] python-docx НЕ установлен — .docx файлы будут падать с ошибкой.\n"
            "       Установи: pip install python-docx",
            flush=True,
        )

    # 2) PyMuPDF (для .pdf)
    try:
        import fitz  # noqa: F401
        print("[ok]   PyMuPDF установлен — .pdf будут индексироваться", flush=True)
    except ImportError:
        print(
            "[WARN] PyMuPDF НЕ установлен — .pdf не будут читаться.\n"
            "       Установи: pip install pymupdf",
            flush=True,
        )

    # 3) Tesseract OCR (опционально — нужен ТОЛЬКО для сканированных PDF)
    try:
        import pytesseract  # noqa: F401
        try:
            ver = pytesseract.get_tesseract_version()
            print(f"[ok]   Tesseract найден ({ver}) — OCR доступен", flush=True)
        except Exception:
            print(
                "[info] Tesseract OCR не установлен (опционально).\n"
                "       Нужен ТОЛЬКО для сканированных PDF без текстового слоя.\n"
                "       Обычные PDF (с выделяемым текстом) работают без него.",
                flush=True,
            )
    except ImportError:
        print(
            "[info] pytesseract не установлен (опционально, для сканированных PDF).",
            flush=True,
        )

    # 4) watchdog (авто-индексация)
    try:
        import watchdog  # noqa: F401
        print("[ok]   watchdog установлен — авто-наблюдение за corpus/", flush=True)
    except ImportError:
        print(
            "[WARN] watchdog НЕ установлен — новые файлы НЕ будут авто-индексироваться.\n"
            "       Установи: pip install watchdog",
            flush=True,
        )

    # 5) LibreOffice (для конвертации .doc — только если такие файлы есть)
    try:
        doc_files = list((_PROJECT_ROOT / "corpus").rglob("*.doc")) if (
            _PROJECT_ROOT / "corpus"
        ).exists() else []
        doc_files = [p for p in doc_files if p.suffix.lower() == ".doc"]
    except Exception:
        doc_files = []
    if doc_files:
        import shutil as _sh
        soffice = _sh.which("soffice") or _sh.which("soffice.exe")
        if not soffice:
            for cand in (
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ):
                if Path(cand).exists():
                    soffice = cand
                    break
        if soffice:
            print(
                f"[ok]   LibreOffice найден — {len(doc_files)} .doc будут "
                f"автоматически сконвертированы в .docx",
                flush=True,
            )
        else:
            print(
                f"[WARN] Найдено {len(doc_files)} файлов .doc, но LibreOffice "
                f"НЕ установлен.\n"
                "       Старые Word-файлы не будут индексироваться.\n"
                "       Установи: https://www.libreoffice.org/download/",
                flush=True,
            )

    # 6) Ollama сервер
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            r.read(1)
        print("[ok]   Ollama сервер отвечает на http://localhost:11434", flush=True)
    except Exception as e:
        print(
            "[WARN] Ollama сервер НЕДОСТУПЕН на http://localhost:11434.\n"
            "       Без него Step D (embeddings) и ответ ИИ работать не будут.\n"
            f"       Запусти Ollama (иконка в трее) или: ollama serve\n"
            f"       Подробности: {type(e).__name__}",
            flush=True,
        )

    print("=" * 60, flush=True)


def main():
    _preflight_checks()
    app = RAGSearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
