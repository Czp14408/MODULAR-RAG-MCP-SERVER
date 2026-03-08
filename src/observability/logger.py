"""Logging utilities for human logs and trace JSONL."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_trace = getattr(record, "trace", None)
        if isinstance(extra_trace, dict):
            payload.update(extra_trace)
        return json.dumps(payload, ensure_ascii=False)


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


def get_trace_logger(
    name: str = "modular_rag_trace",
    log_path: str = "logs/traces.jsonl",
    log_level: str = "INFO",
) -> logging.Logger:
    """返回专用于 trace 持久化的 JSONL logger。"""
    logger_name = f"{name}:{Path(log_path).resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


def write_trace(trace_dict: Dict[str, Any], log_path: str = "logs/traces.jsonl") -> None:
    """将 trace 字典写入 JSON Lines 文件。"""
    logger = get_trace_logger(log_path=log_path)
    logger.info("trace", extra={"trace": trace_dict})
