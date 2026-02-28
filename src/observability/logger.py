"""Minimal logger utility for early project phases."""

import logging
import sys


def get_logger(name: str = "modular_rag_mcp_server", log_level: str = "INFO") -> logging.Logger:
    """Return a stderr logger with a stable formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    return logger
