"""LLM Reranker 的轻量占位实现。"""

from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker, _validate_candidates


class LLMReranker(BaseReranker):
    """按文本长度作为示例分数进行重排序（后续可替换为真实 LLM 打分）。"""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        _validate_candidates(candidates)

        def score(item: Dict[str, Any]) -> float:
            text = str(item.get("text", ""))
            return float(len(text))

        ranked = [dict(item, score=item.get("score", score(item))) for item in candidates]
        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked
