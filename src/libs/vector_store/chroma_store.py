"""ChromaStore 默认后端（轻量可持久化实现）。"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreContractError


class ChromaStore(BaseVectorStore):
    """本地可持久化向量存储实现。

    说明：
    1. 当前阶段使用 JSON 文件模拟持久化，避免引入重依赖。
    2. 保持接口与真实向量库一致：`upsert` + `query` + metadata filters。
    3. 后续切换真实 Chroma 客户端时，可复用同一抽象契约。
    """

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._store_file = self._resolve_store_file()
        self._load_from_disk()

    def upsert(self, records: Iterable[Dict[str, Any]], trace: Optional[Any] = None) -> None:
        changed = False
        for record in records:
            self.validate_record(record)
            self._records[record["id"]] = {
                "id": record["id"],
                "vector": [float(v) for v in record["vector"]],
                "metadata": record.get("metadata", {}) or {},
                "text": record.get("text", ""),
            }
            changed = True

        if changed:
            self._persist_to_disk()

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

    def _resolve_store_file(self) -> Path:
        """解析本地持久化文件路径。"""
        persist_dir = _read_vector_store_option(self.settings, "persist_directory", "data/db/chroma")
        path = Path(str(persist_dir)).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path / "store.json"

    def _load_from_disk(self) -> None:
        """启动时加载历史记录，支持进程重启后的 roundtrip。"""
        if not self._store_file.exists():
            return

        raw = json.loads(self._store_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise VectorStoreContractError("Invalid persisted data: expected list")

        for item in raw:
            self.validate_record(item)
            self._records[item["id"]] = {
                "id": item["id"],
                "vector": [float(v) for v in item["vector"]],
                "metadata": item.get("metadata", {}) or {},
                "text": item.get("text", ""),
            }

    def _persist_to_disk(self) -> None:
        """将当前内存状态落盘。"""
        payload = list(self._records.values())
        self._store_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_vector_store_option(settings: Any, key: str, default: Any) -> Any:
    """从 settings.vector_store 读取配置，兼容 dataclass 与 dict。"""
    if hasattr(settings, "vector_store") and hasattr(settings.vector_store, key):
        value = getattr(settings.vector_store, key)
        return default if value is None else value
    if isinstance(settings, dict):
        vs = settings.get("vector_store")
        if isinstance(vs, dict):
            value = vs.get(key, default)
            return default if value is None else value
    return default


def _match_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """简单等值过滤：metadata[key] == value。"""
    if not filters:
        return True
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算余弦相似度。"""
    if len(vec_a) != len(vec_b):
        raise VectorStoreContractError("Vector dimension mismatch between query and record")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
