"""
Filesystem watcher for the corpus folder.

When any supported file (PDF, TXT, MD, DOCX) is created/modified/moved/deleted
in the corpus, triggers a reindexing pass after a short debounce window
(to batch related events together).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .extractors import supported_extensions


class _CorpusEventHandler:
    """
    Watchdog event handler with debouncing.

    Subclasses FileSystemEventHandler lazily (only when watchdog is installed),
    so importing this module does not require watchdog at runtime.
    """

    def __init__(self, on_change_callback: Callable[[], None], debounce_seconds: float):
        self._callback = on_change_callback
        self._debounce = float(debounce_seconds)
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._supported = {e.lower() for e in supported_extensions()}

    def _is_relevant(self, event) -> bool:
        if getattr(event, "is_directory", False):
            return False
        return Path(event.src_path).suffix.lower() in self._supported

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._callback)
            self._timer.daemon = True
            self._timer.start()

    # Watchdog API hooks
    def on_created(self, event):
        if self._is_relevant(event):
            self._schedule()

    def on_modified(self, event):
        if self._is_relevant(event):
            self._schedule()

    def on_moved(self, event):
        if self._is_relevant(event):
            self._schedule()

    def on_deleted(self, event):
        if self._is_relevant(event):
            self._schedule()


def start_watching(
    corpus_root: Path,
    on_change_callback: Callable[[], None],
    *,
    debounce_seconds: float = 3.0,
):
    """
    Begin recursive monitoring of `corpus_root`. Returns an Observer
    that can be stopped via .stop() + .join().

    Raises ImportError if `watchdog` is not installed.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception as e:
        raise ImportError(
            "watchdog is required for the corpus watcher. "
            "Install with: pip install watchdog"
        ) from e

    handler_logic = _CorpusEventHandler(on_change_callback, debounce_seconds)

    class _Adapter(FileSystemEventHandler):
        def on_created(self, event):
            handler_logic.on_created(event)

        def on_modified(self, event):
            handler_logic.on_modified(event)

        def on_moved(self, event):
            handler_logic.on_moved(event)

        def on_deleted(self, event):
            handler_logic.on_deleted(event)

    observer = Observer()
    observer.schedule(_Adapter(), str(corpus_root), recursive=True)
    observer.start()
    return observer


def _log(msg: str) -> None:
    """Print a progress message and flush so the user sees it in real time."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run_reindex(
    project_root: Path,
    *,
    embedder_model: str = "mxbai-embed-large",
    enable_ocr: bool = True,
    ocr_lang: str = "rus+eng",
) -> None:
    """
    Run Step A (scan) + Step B (extract) + Step D (embed) as subprocesses.

    Output of every subprocess is INHERITED from the parent process — so when
    the desktop app is launched from a .bat, the user sees every line of
    progress in the cmd window.

    OCR is enabled by default — without it scanned PDFs (e.g. image-based
    Chinese/Japanese documents) extract zero text and become unsearchable.
    """
    py = sys.executable
    kwargs = {"cwd": str(project_root)}  # inherit stdout/stderr

    t0 = time.time()
    _log("=== Reindex started ===")

    _log("Step 0: converting legacy .doc files to .docx (if any)...")
    subprocess.run([py, "scripts/convert_doc_to_docx.py"], **kwargs)

    _log("Step A: scanning corpus folder for new / changed files...")
    subprocess.run(
        [py, "scripts/step_a_scan_corpus.py", "--prune-missing"], **kwargs
    )

    _log(
        "Step B: extracting text from PDF / TXT / MD / DOCX"
        + (" (OCR enabled)" if enable_ocr else "")
        + "..."
    )
    step_b = [py, "scripts/step_b_index_pages.py"]
    if enable_ocr:
        step_b += ["--ocr", "--ocr-lang", ocr_lang]
    subprocess.run(step_b, **kwargs)

    _log("Step D: building semantic embeddings via Ollama...")
    subprocess.run(
        [
            py,
            "scripts/step_d_embed_chunks.py",
            "--embedder", "ollama",
            "--ollama-model", embedder_model,
        ],
        **kwargs,
    )

    _log(f"=== Reindex done in {time.time() - t0:.1f}s ===\n")
