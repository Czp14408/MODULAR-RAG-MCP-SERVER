"""TraceCollector：负责将 TraceContext 持久化到 JSONL。"""

from __future__ import annotations

from typing import Any, Dict

from src.observability.logger import write_trace


class TraceCollector:
    """将 trace 对象或字典统一写入日志文件。"""

    def __init__(self, log_path: str = "logs/traces.jsonl") -> None:
        self.log_path = log_path

    def collect(self, trace: Any) -> None:
        if hasattr(trace, "to_dict"):
            payload = trace.to_dict()
        elif isinstance(trace, dict):
            payload = dict(trace)
        else:
            raise TypeError("trace must provide to_dict() or be dict")
        write_trace(payload, log_path=self.log_path)
