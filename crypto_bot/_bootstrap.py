"""Re-launch with project .venv when invoked from conda/system Python."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _ROOT / ".venv" / "bin" / "python"


def ensure_project_venv() -> None:
    """
    If .venv exists and we're not already using it, re-exec with venv Python.

    Avoids NumPy 2.x (conda base) breaking scipy/hmmlearn built for NumPy 1.x.
    """
    if not _VENV_PYTHON.exists():
        print(
            "Project virtualenv not found. Create it with:\n"
            f"  cd {_ROOT}\n"
            "  python3.12 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -e .\n"
        )
        sys.exit(1)

    try:
        same = Path(sys.executable).resolve() == _VENV_PYTHON.resolve()
    except OSError:
        same = False

    if same:
        return

    # Re-run this process with the venv interpreter
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), *sys.argv])
