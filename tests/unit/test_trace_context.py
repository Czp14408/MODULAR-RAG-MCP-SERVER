"""F1: TraceContext 测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.trace import TraceContext


def test_trace_context_finish_and_to_dict_are_json_serializable() -> None:
    trace = TraceContext(trace_type="query")
    trace.record_stage(
        "dense_retrieval",
        elapsed_ms=12.5,
        method="dense_vector_search",
        provider="HashEmbedding",
        result_count=3,
    )
    trace.finish()

    payload = trace.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    print(f"[F1] trace_payload={payload}")

    assert payload["trace_type"] == "query"
    assert payload["finished_at"] is not None
    assert payload["total_elapsed_ms"] >= 0
    assert payload["stages"][0]["method"] == "dense_vector_search"
    assert "trace_id" in payload
    assert encoded
