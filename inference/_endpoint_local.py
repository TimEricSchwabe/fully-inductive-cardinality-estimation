"""Loader for the endpoint-specific action parameter.

The value is read from a local file (``inference/.endpoint_action``). Create the file with a single line containing the action
string before running ``build_ranking_dataset.py`` or ``dp_subquery_eval.py``.
"""
from __future__ import annotations

from pathlib import Path

_LOCAL_FILE = Path(__file__).resolve().parent / ".endpoint_action"


def get_endpoint_action() -> str:
    """Return the endpoint-specific action parameter from the local file.

    Raises FileNotFoundError with a clear message if the file is missing.
    """
    if not _LOCAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {_LOCAL_FILE}. Create it with a single line containing "
            f"the endpoint-specific action parameter."
        )
    value = _LOCAL_FILE.read_text().strip()
    if not value:
        raise ValueError(f"{_LOCAL_FILE} is empty.")
    return value
