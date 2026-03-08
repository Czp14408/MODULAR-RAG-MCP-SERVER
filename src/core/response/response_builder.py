"""ResponseBuilder：构造 MCP tool 响应体。"""

from __future__ import annotations

from typing import Dict, List

from src.core.response.citation_generator import CitationGenerator
from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.types import RetrievalResult


class ResponseBuilder:
    """将 retrieval results 转成 MCP tool result 结构。"""

    def __init__(
        self,
        citation_generator: CitationGenerator | None = None,
        multimodal_assembler: MultimodalAssembler | None = None,
    ) -> None:
        self.citation_generator = citation_generator or CitationGenerator()
        self.multimodal_assembler = multimodal_assembler or MultimodalAssembler()

    def build(self, retrieval_results: List[RetrievalResult], query: str) -> Dict[str, object]:
        if not retrieval_results:
            return {
                "content": [{"type": "text", "text": "未找到相关文档，请先运行 ingest.py 摄取数据。"}],
                "structuredContent": {"query": query, "citations": [], "results": []},
            }

        citations = self.citation_generator.generate(retrieval_results)
        markdown = self._build_markdown(retrieval_results)
        multimodal = self.multimodal_assembler.assemble(retrieval_results)
        return {
            "content": [{"type": "text", "text": markdown}, *multimodal],
            "structuredContent": {
                "query": query,
                "citations": citations,
                "results": [item.to_dict() for item in retrieval_results],
            },
        }

    @staticmethod
    def _build_markdown(retrieval_results: List[RetrievalResult]) -> str:
        lines: List[str] = []
        for index, item in enumerate(retrieval_results, start=1):
            source = str(item.metadata.get("source_path", ""))
            page = item.metadata.get("page", "-")
            preview = item.text.strip().replace("\n", " ")
            preview = preview[:160]
            lines.append(f"[{index}] {preview}")
            lines.append(f"来源: {source} | 页码: {page} | 分数: {item.score:.4f}")
        return "\n\n".join(lines)
