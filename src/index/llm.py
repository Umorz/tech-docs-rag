from __future__ import annotations

"""
LLM client for Query-time step (7): generate a final answer from retrieved context.

Design goals
------------
- stdlib-only (urllib). No new dependencies.
- mirrors the style of src.index.embeddings.OllamaEmbedder so the project stays
  consistent.
- explicit RAG: the prompt always cites sources by bracketed indices like [1], [2].
- safe for non-LLM environments: a missing/unreachable Ollama yields a clear
  LLMError that the caller can show to the user instead of crashing.

This module does NOT know about SQLite, retrieval, or REPL state. It only knows
how to talk to an Ollama chat endpoint and how to build a RAG prompt from a list
of "context items" (text + citation key) that the caller already produced.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence


class LLMError(RuntimeError):
    """User-facing error for LLM-related failures."""


# -----------------------------------------------------------------------------
# Config / data model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaChatConfig:
    model: str = "t-lite-it"
    base_url: str = "http://localhost:11434"
    timeout_s: int = 300
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 1024
    # Optional override for the system prompt (Russian by default).
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class LLMContextItem:
    """
    Minimal shape this module needs from the caller's retrieval results.

    The caller (REPL) keeps richer types (citation, score, channel). For LLM input
    we only need a stable label, the text, and a printable citation string.
    """

    label: str  # e.g. "1", "2"
    citation: str  # e.g. "datasheets/lm317.pdf:12:5-9"
    text: str


# -----------------------------------------------------------------------------
# Default system prompt (RU)
# -----------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT_RU = (
    "Ты — ассистент по поиску в коллекции документов. "
    "Отвечай по-русски, кратко и по делу.\n\n"
    "Правила:\n"
    "1. Используй ТОЛЬКО фрагменты из раздела «Источники» ниже. Не выдумывай факты.\n"
    "2. Если ответа во фрагментах нет — честно скажи, что в найденных документах этого нет.\n"
    "3. Сохраняй имена, обозначения, идентификаторы, даты и числа в исходном виде. "
    "Не «исправляй» их.\n"
    "4. Числовые величины приводи с единицами измерения, как в источнике.\n"
    "5. Если в источнике указан диапазон «X to Y» — приводи его целиком, а не одну границу.\n"
    "6. В конце каждого утверждения ставь номер источника в квадратных скобках: [1], [2].\n"
    "7. Не пересказывай весь источник целиком — только то, что отвечает на вопрос."
)


# -----------------------------------------------------------------------------
# Prompt assembly
# -----------------------------------------------------------------------------


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def build_rag_messages(
    query: str,
    items: Sequence[LLMContextItem],
    *,
    system_prompt: Optional[str] = None,
    max_total_context_chars: int = 12000,
    max_chars_per_item: int = 2000,
) -> list[dict[str, str]]:
    """
    Build chat messages for an Ollama /api/chat call.

    Strategy:
      - hard-cap per-chunk text length so a single long chunk can't eat the budget
      - hard-cap total context characters; drop the tail if needed
      - format every chunk with its bracketed label and a citation string
    """
    sys_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT_RU).strip()

    blocks: list[str] = []
    total = 0
    for it in items:
        body = _truncate((it.text or "").strip(), max_chars_per_item)
        block = f"[{it.label}] {it.citation}\n{body}"
        if total + len(block) > max_total_context_chars and blocks:
            break
        blocks.append(block)
        total += len(block)

    sources_section = "\n\n".join(blocks) if blocks else "(нет найденных фрагментов)"

    user_msg = (
        f"Вопрос: {query.strip()}\n\n"
        f"Источники:\n{sources_section}\n\n"
        "Ответь по правилам из системного сообщения, ссылайся на источники как [1], [2] и т.д."
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]


# -----------------------------------------------------------------------------
# Ollama client
# -----------------------------------------------------------------------------


class OllamaChat:
    """
    Thin client around POST {base_url}/api/chat.

    Endpoint shape (Ollama):
      POST /api/chat
        {
          "model": "...",
          "messages": [{"role": "...", "content": "..."}, ...],
          "stream": true/false,
          "options": {"temperature": ..., "top_p": ..., "num_predict": ...}
        }
      ->
      streaming: a sequence of JSON lines, each {"message": {"role":"assistant","content":"..."}, "done": bool}
      non-streaming: single JSON {"message": {"role":"assistant","content":"..."}, ...}
    """

    def __init__(self, config: Optional[OllamaChatConfig] = None) -> None:
        self._cfg = config or OllamaChatConfig()
        if not self._cfg.model:
            raise ValueError("Ollama chat model must be non-empty")
        if not self._cfg.base_url:
            raise ValueError("Ollama base_url must be non-empty")
        if int(self._cfg.timeout_s) <= 0:
            raise ValueError("Ollama timeout_s must be > 0")

    @property
    def config(self) -> OllamaChatConfig:
        return self._cfg

    # ---- Health check ----

    def ping(self) -> bool:
        """
        Best-effort check that Ollama is up. Returns True if /api/tags responds 2xx.
        """
        url = self._cfg.base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return 200 <= int(resp.status) < 300
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        Return installed model names from /api/tags. Empty list on failure.
        """
        url = self._cfg.base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return []
        try:
            data = json.loads(body)
        except Exception:
            return []
        out: list[str] = []
        for m in data.get("models", []) or []:
            name = m.get("name")
            if isinstance(name, str):
                out.append(name)
        return out

    # ---- Core call ----

    def _payload(self, messages: Sequence[dict[str, str]], stream: bool) -> dict[str, Any]:
        return {
            "model": self._cfg.model,
            "messages": list(messages),
            "stream": bool(stream),
            "options": {
                "temperature": float(self._cfg.temperature),
                "top_p": float(self._cfg.top_p),
                "num_predict": int(self._cfg.num_predict),
            },
        }

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """
        Non-streaming chat. Returns the assistant's full content as a single string.
        """
        url = self._cfg.base_url.rstrip("/") + "/api/chat"
        data = json.dumps(self._payload(messages, stream=False)).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=int(self._cfg.timeout_s)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise LLMError(f"Ollama HTTP error {e.code} for {url}: {raw}") from e
        except Exception as e:
            raise LLMError(f"Failed to call Ollama at {url}: {e}") from e

        try:
            obj = json.loads(body)
        except Exception as e:
            raise LLMError(f"Invalid JSON from Ollama: {body[:2000]!r}") from e

        msg = obj.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise LLMError(f"Ollama response missing message.content: {obj!r}")
        return content

    def chat_stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """
        Streaming chat. Yields content chunks (delta strings) as they arrive.

        Each Ollama stream line is a JSON object; the assistant's incremental tokens
        live in obj["message"]["content"]. The final line has "done": true.
        """
        url = self._cfg.base_url.rstrip("/") + "/api/chat"
        data = json.dumps(self._payload(messages, stream=True)).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=int(self._cfg.timeout_s))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise LLMError(f"Ollama HTTP error {e.code} for {url}: {raw}") from e
        except Exception as e:
            raise LLMError(f"Failed to open stream to Ollama at {url}: {e}") from e

        try:
            for raw_line in resp:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # Skip non-JSON lines defensively (shouldn't happen in /api/chat).
                    continue

                msg = obj.get("message") or {}
                piece = msg.get("content")
                if isinstance(piece, str) and piece:
                    yield piece

                if obj.get("done") is True:
                    break
        finally:
            try:
                resp.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Convenience: run RAG end-to-end
# -----------------------------------------------------------------------------


def answer_with_context(
    client: OllamaChat,
    query: str,
    items: Sequence[LLMContextItem],
    *,
    system_prompt: Optional[str] = None,
    max_total_context_chars: int = 12000,
    max_chars_per_item: int = 2000,
    stream_writer: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Build a RAG prompt, send to Ollama, and return the full answer text.

    If `stream_writer` is provided, deltas are written to it as they arrive
    (e.g. `sys.stdout.write`), and the function still returns the full text.

    Raises LLMError on any backend failure.
    """
    messages = build_rag_messages(
        query,
        items,
        system_prompt=system_prompt,
        max_total_context_chars=max_total_context_chars,
        max_chars_per_item=max_chars_per_item,
    )

    if stream_writer is None:
        return client.chat(messages)

    parts: list[str] = []
    for delta in client.chat_stream(messages):
        parts.append(delta)
        try:
            stream_writer(delta)
        except Exception:
            # Streaming UI shouldn't break the answer assembly.
            pass
    return "".join(parts)
