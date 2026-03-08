"""TraceService：读取 JSONL trace 并提供过滤能力。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TraceService:
    """读取 traces.jsonl，供 Dashboard 追踪页使用。"""

    def __init__(self, trace_log_path: str = "logs/traces.jsonl") -> None:
        self.trace_log_path = Path(trace_log_path)

    def list_traces(self, trace_type: Optional[str] = None) -> List[Dict[str, Any]]:
        traces = self._load_all()
        if trace_type:
            traces = [item for item in traces if item.get("trace_type") == trace_type]
        traces.sort(key=lambda item: str(item.get("started_at", "")), reverse=True)
        return traces

    def search_query_traces(self, keyword: str = "") -> List[Dict[str, Any]]:
        traces = self.list_traces(trace_type="query")
        if not keyword.strip():
            return traces
        keyword = keyword.strip().lower()
        return [
            item
            for item in traces
            if keyword in str(self._extract_query_text(item)).lower()
        ]

    def _load_all(self) -> List[Dict[str, Any]]:
        if not self.trace_log_path.exists():
            return []
        traces: List[Dict[str, Any]] = []
        for line in self.trace_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            traces.append(json.loads(line))
        return traces

    @staticmethod
    def _extract_query_text(trace: Dict[str, Any]) -> str:
        for stage in trace.get("stages", []):
            if stage.get("stage") == "query_processing":
                return str(stage.get("query", ""))
        return ""
