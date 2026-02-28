"""A2 smoke test: verify top-level package imports."""

import importlib
import unittest


class TestSmokeImports(unittest.TestCase):
    """Import smoke tests for top-level packages."""

    def test_top_level_package_imports(self) -> None:
        for package in ("mcp_server", "core", "ingestion", "libs", "observability"):
            module = importlib.import_module(package)
            self.assertIsNotNone(module)
