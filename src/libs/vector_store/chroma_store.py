"""ChromaStore 的内存版桩实现（用于阶段 B4 契约验证）。"""

from math import sqrt
from typing import Any, Dict, Iterable, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreContractError


class ChromaStore(BaseVectorStore):
    """不依赖真实数据库的最小可运行向量存储实现。"""

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._records: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: Iterable[Dict[str, Any]], trace: Optional[Any] = None) -> None:
        for record in records:
            self.validate_record(record)
            self._records[record["id"]] = {
                "id": record["id"],
                "vector": [float(v) for v in record["vector"]],
                "metadata": record.get("metadata", {}) or {},
                "text": record.get("text", ""),
            }

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        self.validate_vector(vector)
        if not isinstance(top_k, int) or top_k <= 0:
            raise VectorStoreContractError("Invalid top_k: must be positive int")
        if filters is not None and not isinstance(filters, dict):
            raise VectorStoreContractError("Invalid filters: must be dict")

        query_vec = [float(v) for v in vector]
        candidates = []
        for record in self._records.values():
            if not _match_filters(record.get("metadata", {}), filters):
                continue
            score = _cosine_similarity(query_vec, record["vector"])
            candidates.append(
                {
                    "id": record["id"],
                    "score": score,
                    "metadata": record.get("metadata", {}),
                    "text": record.get("text", ""),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]


def _match_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """简单等值过滤：metadata[key] == value。"""
    if not filters:
        return True
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算余弦相似度，向量维度不一致时抛出契约错误。"""
    if len(vec_a) != len(vec_b):
        raise VectorStoreContractError("Vector dimension mismatch between query and record")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
