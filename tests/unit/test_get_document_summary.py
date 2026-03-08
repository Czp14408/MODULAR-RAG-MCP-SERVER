"""E5: get_document_summary tool 测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_server.tools.get_document_summary import get_document_summary


def test_get_document_summary_returns_structured_payload(tmp_path: Path) -> None:
    """存在 doc_id 时应返回标题、摘要、标签等结构化字段。"""
    store_file = tmp_path / "store.json"
    store_file.write_text(
        json.dumps(
            [
                {
                    "id": "chunk-1",
                    "metadata": {
                        "document_id": "doc-1",
                        "source_path": "tests/data/test_chunking_text.pdf",
                        "title": "文档标题",
                        "summary": "这里是一段摘要。",
                        "tags": ["rag", "pdf"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    result = get_document_summary({"doc_id": "doc-1"}, {"vector_store_file": str(store_file)})
    print(f"[E5] summary_result={result}")

    assert result["structuredContent"]["doc_id"] == "doc-1"
    assert result["structuredContent"]["title"] == "文档标题"
    assert result["structuredContent"]["summary"] == "这里是一段摘要。"
    assert result["structuredContent"]["tags"] == ["rag", "pdf"]


def test_get_document_summary_raises_for_unknown_doc_id(tmp_path: Path) -> None:
    """不存在的 doc_id 必须返回规范错误，而不是空结果。"""
    store_file = tmp_path / "store.json"
    store_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="document not found: missing-doc"):
        get_document_summary({"doc_id": "missing-doc"}, {"vector_store_file": str(store_file)})
