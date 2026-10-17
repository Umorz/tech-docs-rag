from __future__ import annotations

"""
Project entrypoint: interactive REPL.

This file intentionally does NOT replace or modify any of the existing step scripts:
- scripts/step_a_scan_corpus.py
- scripts/step_b_index_pages.py
- scripts/step_c_query.py
- scripts/step_d_embed_chunks.py

It simply provides a convenient user-facing entrypoint:
  python Main.py

Later (when adding an LLM), the REPL can grow new commands without breaking the
existing indexing/retrieval pipeline.
"""

import sys
from pathlib import Path


def _configure_utf8_stdio() -> None:
    """
    Configure UTF-8 stdout/stderr to avoid Windows console encoding errors (e.g. cp1251)
    when printing paths or text that contains non-ASCII characters.

    We use errors="replace" to ensure the REPL never crashes due to encoding issues.
    """
    # Python 3.7+: TextIOWrapper supports .reconfigure()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _ensure_project_root_on_syspath() -> None:
    # Make imports like `from src.app.repl import main` work regardless of CWD.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _ensure_project_root_on_syspath()
    _configure_utf8_stdio()

    from src.app.repl import main as repl_main  # local import after sys.path tweak

    return int(repl_main())


if __name__ == "__main__":
    raise SystemExit(main())
