"""Startup warning filters for direct script execution from ``scripts/``."""

from __future__ import annotations

import warnings

try:
    from urllib3.exceptions import NotOpenSSLWarning  # type: ignore

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass
