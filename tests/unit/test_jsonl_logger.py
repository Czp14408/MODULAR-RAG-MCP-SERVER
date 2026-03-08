"""F2: JSONL trace logger 测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observability.logger import write_trace


def test_write_trace_appends_one_json_line(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    trace = {
        "trace_id": "trace-1",
        "trace_type": "ingestion",
        "started_at": "2026-03-08T00:00:00+00:00",
        "finished_at": "2026-03-08T00:00:01+00:00",
        "total_elapsed_ms": 1000.0,
        "stages": [],
    }

    write_trace(trace, log_path=str(log_path))
    lines = log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    print(f"[F2] trace_line={payload}")

    assert len(lines) == 1
    assert payload["trace_type"] == "ingestion"
    assert payload["trace_id"] == "trace-1"
