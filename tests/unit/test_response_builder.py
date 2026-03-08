"""E3: ResponseBuilder 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.response.response_builder import ResponseBuilder
from src.core.types import RetrievalResult


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        score=0.9,
        text="这是命中的文本片段。",
        metadata={
            "source_path": "tests/data/test_chunking_multimodal.pdf",
            "page": 1,
        },
    )


def test_response_builder_builds_markdown_and_citations() -> None:
    builder = ResponseBuilder()
    result = builder.build([_result()], query="测试查询")
    print(f"[E3] response={result}")

    assert result["content"][0]["type"] == "text"
    assert "[1]" in result["content"][0]["text"]
    citations = result["structuredContent"]["citations"]
    assert citations[0]["source"] == "tests/data/test_chunking_multimodal.pdf"
    assert citations[0]["chunk_id"] == "chunk-1"


def test_response_builder_handles_empty_results_gracefully() -> None:
    builder = ResponseBuilder()
    result = builder.build([], query="空查询")
    assert "未找到相关文档" in result["content"][0]["text"]
