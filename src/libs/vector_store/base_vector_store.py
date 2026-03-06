"""VectorStore 抽象契约。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


class VectorStoreContractError(ValueError):
    """向量存储契约不满足时抛出的错误。"""


class BaseVectorStore(ABC):
    """向量库统一抽象接口。"""

    def __init__(self, settings: Any) -> None:
        # 保留配置对象，便于具体实现读取 provider 参数。
        self.settings = settings

    @abstractmethod
    def upsert(self, records: Iterable[Dict[str, Any]], trace: Optional[Any] = None) -> None:
        """写入或更新记录。"""

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """按向量检索并返回 top-k 结果。"""

    @abstractmethod
    def get_by_ids(self, ids: List[str], trace: Optional[Any] = None) -> List[Dict[str, Any]]:
        """根据 chunk id 批量获取记录。"""

    @staticmethod
    def validate_vector(vector: List[float], field: str = "vector") -> None:
        """校验向量 shape。"""
        if not isinstance(vector, list) or not vector:
            raise VectorStoreContractError(f"Invalid {field}: must be non-empty list[float]")
        for value in vector:
            if not isinstance(value, (int, float)):
                raise VectorStoreContractError(f"Invalid {field}: all elements must be numeric")

    @staticmethod
    def validate_record(record: Dict[str, Any]) -> None:
        """校验单条 upsert 记录 shape。"""
        if not isinstance(record, dict):
            raise VectorStoreContractError("Invalid record: must be dict")
        if not isinstance(record.get("id"), str) or not record["id"].strip():
            raise VectorStoreContractError("Invalid record.id: must be non-empty string")
        BaseVectorStore.validate_vector(record.get("vector"), field="record.vector")
        metadata = record.get("metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise VectorStoreContractError("Invalid record.metadata: must be dict")
