"""Main entry for Modular RAG MCP Server."""

import sys
from pathlib import Path

from src.core.settings import SettingsError, load_settings
from src.observability.logger import get_logger


def main() -> int:
    """Load settings and initialize core runtime wiring."""
    settings_path = Path("config/settings.yaml")
    try:
        settings = load_settings(settings_path)
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = get_logger(log_level=settings.observability.log_level)
    logger.info("Settings loaded successfully from %s", settings_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
