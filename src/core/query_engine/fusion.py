"""Fusion：RRF 融合 dense/sparse 排名。"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.types import RetrievalResult


class Fusion:
    """使用 Reciprocal Rank Fusion 合并多路检索结果。"""

    def __init__(self, k: int = 60) -> None:
        self.k = max(1, int(k))

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        score_map: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        for results in (dense_results, sparse_results):
            for rank, item in enumerate(results, start=1):
                score_map[item.chunk_id] = score_map.get(item.chunk_id, 0.0) + 1.0 / (self.k + rank)
                result_map.setdefault(item.chunk_id, item)

        ranked_ids = sorted(score_map.keys(), key=lambda cid: (-score_map[cid], cid))
        fused = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=score_map[chunk_id],
                text=result_map[chunk_id].text,
                metadata=dict(result_map[chunk_id].metadata),
            )
            for chunk_id in ranked_ids
        ]
        return fused[:top_k] if top_k is not None else fused
