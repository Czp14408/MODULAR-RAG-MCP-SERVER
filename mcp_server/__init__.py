"""Compatibility wrapper for importing `mcp_server` from repo root."""
from importlib import import_module as _import_module
_pkg = _import_module("src.mcp_server")
globals().update(_pkg.__dict__)
__path__ = _pkg.__path__
