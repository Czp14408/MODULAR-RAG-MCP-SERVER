"""VectorUpserter：生成稳定 ID 并写入向量库。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import ChunkRecord
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """对 DenseEncoder 输出执行幂等 upsert。"""

    def __init__(self, settings: Any, vector_store: Optional[BaseVectorStore] = None) -> None:
        self.settings = settings
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def upsert(self, records: List[ChunkRecord], trace: Optional[TraceContext] = None) -> List[ChunkRecord]:
        stable_records: List[ChunkRecord] = []
        payload = []

        for record in records:
            stable_id = self._generate_chunk_id(record)
            stable_record = replace(record, id=stable_id)
            stable_records.append(stable_record)
            payload.append(
                {
                    "id": stable_id,
                    "vector": list(stable_record.dense_vector or []),
                    "metadata": dict(stable_record.metadata),
                    "text": stable_record.text,
                }
            )

        self.vector_store.upsert(payload, trace=trace)
        if trace is not None:
            trace.record_stage(
                "vector_upserter",
                elapsed_ms=0.0,
                record_count=len(stable_records),
            )
        return stable_records

    @staticmethod
    def _generate_chunk_id(record: ChunkRecord) -> str:
        source_path = str(record.metadata.get("source_path", ""))
        chunk_index = str(record.metadata.get("chunk_index", ""))
        content_hash = hashlib.md5(record.text.encode("utf-8")).hexdigest()[:8]  # noqa: S324
        # 参数选择说明：
        # 直接拼接 source_path + chunk_index + content_hash，满足 spec 的确定性要求，
        # 同时保留可读性，便于后续问题排查。
        return f"{source_path}:{chunk_index}:{content_hash}"
