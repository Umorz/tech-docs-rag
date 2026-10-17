"""
Streamlit-based web GUI for the RAG search system.

Запуск:
    pip install streamlit
    streamlit run src/app/web.py

Модуль независим от REPL-приложения. Использует только базовые модули
из src/index/: db, embeddings, hybrid, search, llm.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Подключаем корень проекта к sys.path, чтобы работали `from src...`-импорты
# при запуске через `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.index.db import connect
from src.index.embeddings import EmbeddingsError, create_embedder
from src.index.hybrid import HybridOptions, hybrid_search_rrf, semantic_retrieve_sqlite
from src.index.llm import (
    LLMContextItem,
    LLMError,
    OllamaChat,
    OllamaChatConfig,
    build_rag_messages,
)
from src.index.search import SearchOptions, search


DB_PATH = _PROJECT_ROOT / "data" / "index.sqlite"


# -----------------------------
# Page configuration & styling
# -----------------------------

st.set_page_config(
    page_title="Универсальный поиск",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #1a73e8;
        font-size: 2.4em;
        font-weight: 500;
        margin-top: 0.3em;
        margin-bottom: 0.2em;
    }
    .main-subtitle {
        text-align: center;
        color: #5f6368;
        font-size: 1em;
        margin-bottom: 1.5em;
    }
    .ai-overview {
        background: linear-gradient(135deg, #e8f0fe 0%, #f8fbff 100%);
        border-left: 4px solid #1a73e8;
        padding: 1.2em 1.5em;
        border-radius: 0.6em;
        margin: 1em 0 1.5em 0;
    }
    .ai-overview-title {
        color: #1a73e8;
        font-weight: 600;
        font-size: 1.15em;
        margin-bottom: 0.6em;
    }
    .result-block {
        padding: 0.8em 0;
        border-bottom: 1px solid #ebebeb;
    }
    .result-title {
        color: #1a0dab;
        font-size: 1.15em;
        font-weight: 500;
        margin-bottom: 0.15em;
    }
    .result-citation {
        color: #006621;
        font-size: 0.9em;
        margin-bottom: 0.4em;
    }
    .result-snippet {
        color: #4d5156;
        font-size: 0.97em;
        line-height: 1.55;
        white-space: pre-wrap;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------


def _has_cyrillic(text: str, min_chars: int = 2) -> bool:
    n = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return n >= min_chars


@st.cache_resource(show_spinner=False)
def get_embedder():
    """Cached ollama embedder for semantic retrieval."""
    return create_embedder(
        preferred=["ollama"],
        ollama_model="mxbai-embed-large",
        normalized=True,
    )


@st.cache_resource(show_spinner=False)
def get_llm_client(model: str):
    """Cached Ollama chat client (per model name)."""
    return OllamaChat(OllamaChatConfig(model=model, temperature=0.2))


@st.cache_data(ttl=60, show_spinner=False)
def get_available_chat_models() -> list[str]:
    """
    Запрашивает список установленных моделей у Ollama и фильтрует те, что
    подходят для чат-режима. Эмбеддинг-модели (nomic, mxbai, bge) исключены.

    Кешируется на 60 секунд, чтобы не дёргать сервер на каждый ререндер.
    Возвращает значение по умолчанию, если Ollama недоступна.
    """
    fallback = ["t-pro-it"]
    try:
        probe = OllamaChat(OllamaChatConfig(model="probe"))
        if not probe.ping():
            return fallback
        all_models = probe.list_models()
    except Exception:
        return fallback

    embedding_kw = ("embed", "nomic", "mxbai", "bge")
    chat = [m for m in all_models if not any(k in m.lower() for k in embedding_kw)]

    def _sort_key(name: str) -> tuple[int, str]:
        n = name.lower()
        if "t-pro" in n:
            return (0, n)
        if "t-lite" in n:
            return (1, n)
        return (2, n)

    chat.sort(key=_sort_key)
    return chat or fallback


@st.cache_data(show_spinner=False)
def translate_query(query: str, llm_model: str) -> str | None:
    """
    Translate a Russian query to English via Ollama chat.
    Returns None if translation is impossible or not needed.
    Cached per (query, model) pair within the session.
    """
    if not _has_cyrillic(query):
        return None
    client = get_llm_client(llm_model)
    if not client.ping():
        return None
    sys_prompt = (
        "You translate Russian search queries into concise English search queries. "
        "Output ONLY the English query, on a single line, without quotes or "
        "explanations. Keep names, identifiers, dates, numbers and units "
        "unchanged. Keep already-English terms as-is. Use natural, neutral "
        "English suitable for general document search."
    )
    try:
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
    for label in ("translation:", "english:", "the translation is:"):
        if out.lower().startswith(label):
            out = out[len(label):].strip().strip('"').strip("'").strip()
            break
    if not out or len(out) > 500 or _has_cyrillic(out, 4):
        return None
    return out


def _retrieve(mode: str, query: str, limit: int) -> list[dict]:
    """Run retrieval in the chosen mode. Returns a list of result dicts."""
    conn = connect(DB_PATH)
    try:
        if mode == "lexical":
            _, hits = search(
                conn, query, options=SearchOptions(target="lines", limit=limit)
            )
            return [
                {
                    "rel_path": h.rel_path,
                    "page_no": int(h.page_no),
                    "start_line": int(h.line_no),
                    "end_line": int(h.line_no),
                    "text": h.text or h.snippet or "",
                    "score": float(h.score),
                }
                for h in hits
            ]

        embedder = get_embedder()
        if mode == "semantic":
            _, cands = semantic_retrieve_sqlite(
                conn, embedder=embedder, query=query, top_k=limit
            )
            return [
                {
                    "rel_path": c.key.rel_path,
                    "page_no": int(c.key.page_no),
                    "start_line": int(c.key.start_line_no),
                    "end_line": int(c.key.end_line_no),
                    "text": c.text or "",
                    "score": float(c.score),
                }
                for c in cands
            ]

        # hybrid
        _, fused = hybrid_search_rrf(
            conn,
            query,
            lexical_options=SearchOptions(target="lines"),
            embedder=embedder,
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


def _expand_for_llm(results: list[dict], top_n: int, expand_lines: int) -> list[dict]:
    """
    Expand top-N results by +/- expand_lines on the same page (for LLM context).
    Returns dicts with key 'text_expanded' and updated line range.
    """
    conn = connect(DB_PATH)
    try:
        out = []
        for r in results[:top_n]:
            expanded_start = max(1, int(r["start_line"]) - expand_lines)
            expanded_end = int(r["end_line"]) + expand_lines
            rows = conn.execute(
                """
                SELECT l.line_no, l.text
                FROM lines l
                JOIN pages p ON p.page_id = l.page_id
                JOIN documents d ON d.doc_id = p.doc_id
                WHERE d.rel_path = ? AND p.page_no = ?
                  AND l.line_no BETWEEN ? AND ?
                ORDER BY l.line_no
                """,
                (r["rel_path"], r["page_no"], expanded_start, expanded_end),
            ).fetchall()
            if rows:
                text = "\n".join(str(t) for _, t in rows)
                actual_start = int(rows[0][0])
                actual_end = int(rows[-1][0])
            else:
                text = r["text"]
                actual_start = int(r["start_line"])
                actual_end = int(r["end_line"])
            out.append(
                {
                    **r,
                    "text_expanded": text,
                    "start_line_expanded": actual_start,
                    "end_line_expanded": actual_end,
                }
            )
        return out
    finally:
        conn.close()


# -----------------------------
# Default settings (inline controls under the search bar
# replace the previous sidebar panel).
# -----------------------------

mode = "hybrid"
enable_translate = True
enable_expand = True
enable_llm = True
llm_top_n = 6


# -----------------------------
# Sidebar — список документов корпуса
# -----------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _list_indexed_docs() -> list[tuple[str, int, int]]:
    """Returns list of (rel_path, pages_count, status_indexed) from DB."""
    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT d.rel_path,
                   (SELECT COUNT(*) FROM pages p WHERE p.doc_id = d.doc_id) AS pages_n,
                   d.status
            FROM documents d
            ORDER BY d.rel_path
            """
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], int(r[1] or 0), str(r[2] or "")) for r in rows]


with st.sidebar:
    st.markdown("### 📁 Коллекция файлов")
    _docs = _list_indexed_docs()
    if not _docs:
        st.caption("База пуста")
    else:
        for _rel_path, _pages_n, _status in _docs:
            _icon = "📄" if _status == "indexed" else "⚠️"
            st.markdown(
                f'<div style="font-size:0.85em; margin: 0.3em 0;">'
                f'{_icon} {_rel_path}<br>'
                f'<span style="color:#888; font-size:0.85em;">'
                f'{_pages_n} стр., статус: {_status}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Всего: {len(_docs)} документов")


# -----------------------------
# Main UI
# -----------------------------

st.markdown(
    '<div class="main-title">🔎 Универсальный поиск</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    "Гибридный поиск по коллекции файлов с генерацией ответов через языковую модель"
    "</div>",
    unsafe_allow_html=True,
)

query = st.text_input(
    "Поисковый запрос",
    placeholder="Введите запрос на любом языке",
    label_visibility="collapsed",
    key="search_query",
)

# Compact inline controls under the search bar (Perplexity-style).
_c1, _c2, _c3 = st.columns([1, 2, 4])
with _c1:
    limit = st.number_input(
        "Результатов",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
    )
with _c2:
    _available_models = get_available_chat_models()
    _default_idx = 0
    for _i, _m in enumerate(_available_models):
        if "t-pro" in _m.lower():
            _default_idx = _i
            break
    llm_model = st.selectbox(
        "Модель",
        options=_available_models,
        index=_default_idx,
    )


if query:
    # 1. Translate if needed
    effective_query = query
    if enable_translate and enable_llm and _has_cyrillic(query):
        with st.spinner("Перевод запроса..."):
            translated = translate_query(query, llm_model)
        if translated:
            st.markdown(
                f'<div style="color:#9aa0a6; font-size:0.82em; '
                f'margin: 0.4em 0 1em 0; font-style: italic;">'
                f"Запрос для поиска: {translated}"
                f"</div>",
                unsafe_allow_html=True,
            )
            effective_query = translated

    # 2. Retrieve
    with st.spinner("Поиск..."):
        try:
            results = _retrieve(mode, effective_query, limit)
        except EmbeddingsError as e:
            st.error(f"Ошибка модели эмбеддингов: {e}")
            results = []
        except Exception as e:
            st.error(f"Ошибка поиска: {type(e).__name__}: {e}")
            results = []

    # 3. Search results — показываются СРАЗУ после поиска (до запуска ИИ).
    if results:
        st.markdown(
            f"### 📑 Результаты поиска "
            f"<span style='color:#5f6368; font-weight:400; font-size:0.7em;'>"
            f"найдено {len(results)} фрагментов"
            f"</span>",
            unsafe_allow_html=True,
        )

        for i, r in enumerate(results, 1):
            snippet = r["text"]
            if len(snippet) > 500:
                snippet = snippet[:500].rstrip() + "…"
            snippet_safe = (
                snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            st.markdown(
                f"""
                <div class="result-block">
                    <div class="result-title">[{i}] 📄 {r['rel_path']}</div>
                    <div class="result-citation">
                        страница {r['page_no']}, строки {r['start_line']}–{r['end_line']}
                        &nbsp;·&nbsp;
                        релевантность {r['score']:.4f}
                    </div>
                    <div class="result-snippet">{snippet_safe}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("По запросу ничего не найдено. Попробуйте переформулировать.")

    # 4. AI overview — генерируется ПОСЛЕ показа результатов, стримится вниз страницы.
    if enable_llm and results:
        st.markdown('<div class="ai-overview">', unsafe_allow_html=True)
        st.markdown(
            '<div class="ai-overview-title">⚡ Ответ от ИИ</div>',
            unsafe_allow_html=True,
        )

        # Optionally expand context for LLM
        if enable_expand:
            with st.spinner("Расширение контекста..."):
                llm_results = _expand_for_llm(results, llm_top_n, expand_lines=15)
        else:
            llm_results = [
                {**r, "text_expanded": r["text"],
                 "start_line_expanded": r["start_line"],
                 "end_line_expanded": r["end_line"]}
                for r in results[:llm_top_n]
            ]

        client = get_llm_client(llm_model)
        if not client.ping():
            st.warning(
                "⚠️ Сервер Ollama недоступен. "
                "Запустите его и обновите страницу."
            )
        else:
            llm_items = [
                LLMContextItem(
                    label=str(i + 1),
                    citation=(
                        f"{r['rel_path']}, стр. {r['page_no']}, "
                        f"строки {r['start_line_expanded']}–{r['end_line_expanded']}"
                    ),
                    text=r["text_expanded"],
                )
                for i, r in enumerate(llm_results)
            ]
            messages = build_rag_messages(query, llm_items, max_chars_per_item=4000)

            placeholder = st.empty()
            full_answer = ""
            try:
                for delta in client.chat_stream(messages):
                    full_answer += delta
                    placeholder.markdown(full_answer + " ▌")
                placeholder.markdown(full_answer)
            except LLMError as e:
                st.error(f"Ошибка языковой модели: {e}")
            except Exception as e:
                st.error(f"Ошибка: {type(e).__name__}: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# Пустое состояние — без подсказок.
