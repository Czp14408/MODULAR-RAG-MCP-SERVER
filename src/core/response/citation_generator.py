"""CitationGenerator：从 RetrievalResult 构造结构化引用。"""

from __future__ import annotations

from typing import Dict, List

from src.core.types import RetrievalResult


class CitationGenerator:
    """生成 MCP tool 返回中的 citations。"""

    def generate(self, retrieval_results: List[RetrievalResult]) -> List[Dict[str, object]]:
        citations: List[Dict[str, object]] = []
        for item in retrieval_results:
            citations.append(
                {
                    "source": str(item.metadata.get("source_path", "")),
                    "page": item.metadata.get("page"),
                    "chunk_id": item.chunk_id,
                    "score": float(item.score),
                }
            )
        return citations
