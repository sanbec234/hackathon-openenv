"""Compatibility wrapper for validators expecting server/app.py."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is importable before resolving `app`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app as root_app, main as root_main  # re-export existing FastAPI app and runner

app = root_app


def main() -> None:
    """Entry point expected by multi-mode validators."""
    root_main()


if __name__ == "__main__":
    main()
