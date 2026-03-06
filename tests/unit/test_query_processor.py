"""D1: QueryProcessor 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.query_processor import QueryProcessor


def test_query_processor_extracts_non_empty_keywords_and_filters() -> None:
    processor = QueryProcessor()
    result = processor.process("RAG 系统中的向量检索如何工作？", filters={"collection": "demo"})
    print(f"[D1] processed_query={result}")

    assert result.query == "RAG 系统中的向量检索如何工作？"
    assert result.keywords
    assert isinstance(result.filters, dict)
    assert result.filters["collection"] == "demo"
