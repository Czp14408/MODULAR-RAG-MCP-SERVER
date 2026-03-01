"""Cross-Encoder Reranker 的轻量占位实现。"""

from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker, _validate_candidates


class CrossEncoderReranker(BaseReranker):
    """示例实现：优先沿用已有 score，其次按文本长度评分。"""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        _validate_candidates(candidates)

        ranked = []
        for item in candidates:
            if "score" in item:
                score = float(item["score"])
            else:
                score = float(len(str(item.get("text", ""))))
            ranked.append(dict(item, score=score))

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked
