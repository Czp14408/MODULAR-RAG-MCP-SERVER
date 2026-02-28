"""Compatibility wrapper for importing `ingestion` from repo root."""
from importlib import import_module as _import_module
_pkg = _import_module("src.ingestion")
globals().update(_pkg.__dict__)
__path__ = _pkg.__path__
