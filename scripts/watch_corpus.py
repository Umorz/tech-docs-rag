"""
Standalone corpus watcher.

Monitors the corpus folder and automatically reindexes the database whenever
files are added, modified, moved or deleted. Uses a short debounce window
(3 seconds) so a batch of related events triggers a single reindex.

Run from project root:
    pip install watchdog
    python scripts/watch_corpus.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index.watcher import run_reindex, start_watching


def main() -> int:
    corpus = _PROJECT_ROOT / "corpus"
    if not corpus.exists():
        print(f"[error] Corpus folder not found: {corpus}", file=sys.stderr)
        return 1

    print(f"Watching: {corpus}")
    print("Reindex triggers after 3s of inactivity. Press Ctrl+C to stop.")
    print()

    def on_change() -> None:
        print("\n[watch] Detected changes — reindexing...")
        try:
            run_reindex(_PROJECT_ROOT)
        except Exception as e:
            print(f"[watch] Reindex error: {type(e).__name__}: {e}", file=sys.stderr)
        else:
            print("[watch] Done. Waiting for next change.\n")

    try:
        observer = start_watching(corpus, on_change, debounce_seconds=3.0)
    except ImportError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
        observer.join()
        print("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
