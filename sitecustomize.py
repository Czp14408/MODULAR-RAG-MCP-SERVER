"""Interpreter startup hooks for local developer ergonomics.

This module is auto-imported by Python's site initialization when the project
root is on ``sys.path``. We use it to silence one known environment-specific
warning that is unrelated to project behavior.
"""

from __future__ import annotations

import warnings

try:
    from urllib3.exceptions import NotOpenSSLWarning  # type: ignore

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass
