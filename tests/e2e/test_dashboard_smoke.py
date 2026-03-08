"""I2: Dashboard 冒烟测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _prepare_dashboard_workspace(tmp_path: Path) -> dict:
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data" / "db" / "chroma").mkdir(parents=True)
    (tmp_path / "data" / "db" / "bm25").mkdir(parents=True)
    (tmp_path / "data" / "db").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)

    (tmp_path / "config" / "settings.yaml").write_text(
        "\n".join(
            [
                "llm:",
                "  provider: placeholder",
                "embedding:",
                "  provider: hash",
                "vector_store:",
                "  provider: chroma",
                "retrieval:",
                "  top_k: 5",
                "splitter:",
                "  provider: recursive",
                "  chunk_size: 120",
                "  chunk_overlap: 20",
                "rerank:",
                "  enabled: false",
                "evaluation:",
                "  enabled: false",
                "observability:",
                "  log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "db" / "chroma" / "store.json").write_text(
        json.dumps(
            [
                {
                    "id": "chunk-1",
                    "vector": [1.0, 0.0],
                    "text": "alpha",
                    "metadata": {
                        "source_path": "docs/a.pdf",
                        "collection": "demo",
                        "document_id": "doc-a",
                        "title": "A",
                        "summary": "sum-a",
                        "tags": ["a"]
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "logs" / "traces.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"trace_id": "t1", "trace_type": "ingestion", "started_at": "2026-03-08T00:00:00+00:00", "stages": [{"stage": "load", "elapsed_ms": 1.0}]}),
                json.dumps({"trace_id": "t2", "trace_type": "query", "started_at": "2026-03-08T00:00:01+00:00", "stages": [{"stage": "query_processing", "elapsed_ms": 1.0, "query": "demo"}]}),
            ]
        ),
        encoding="utf-8",
    )
    return {
        "DASHBOARD_SETTINGS_PATH": str(tmp_path / "config" / "settings.yaml"),
        "DASHBOARD_DATA_ROOT": str(tmp_path / "data"),
        "DASHBOARD_TRACE_LOG": str(tmp_path / "logs" / "traces.jsonl"),
    }


def test_dashboard_pages_render_without_exception(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    env = _prepare_dashboard_workspace(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    for page_name, expected_title in [
        ("Overview", "System Overview"),
        ("Data Browser", "Data Browser"),
        ("Ingestion Manager", "Ingestion Manager"),
        ("Ingestion Traces", "Ingestion Traces"),
        ("Query Traces", "Query Traces"),
        ("Evaluation Panel", "Evaluation Panel"),
    ]:
        monkeypatch.setenv("DASHBOARD_TEST_PAGE", page_name)
        at = AppTest.from_file(str(PROJECT_ROOT / "src" / "observability" / "dashboard" / "app.py"))
        at.run(timeout=10)
        assert not at.exception
        assert at.title[0].value == expected_title
