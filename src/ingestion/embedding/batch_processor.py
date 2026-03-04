"""BatchProcessor：按批次驱动 dense/sparse 编码。"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, List, Optional, Sequence

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """将 chunk 列表分批处理，保证顺序稳定。"""

    def __init__(
        self,
        settings: Any,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: Optional[int] = None,
    ) -> None:
        self.settings = settings
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = max(1, int(batch_size or self._read_batch_size(settings)))

    def process(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[ChunkRecord]:
        all_records: List[ChunkRecord] = []
        started = perf_counter()

        for batch_index, batch in enumerate(self._iter_batches(chunks, self.batch_size)):
            batch_started = perf_counter()
            dense_records = self.dense_encoder.encode(batch, trace=trace)
            sparse_records = self.sparse_encoder.encode(batch, trace=trace)
            merged = self._merge_records(dense_records, sparse_records)
            all_records.extend(merged)

            if trace is not None:
                trace.record_stage(
                    "batch_processor_batch",
                    elapsed_ms=(perf_counter() - batch_started) * 1000,
                    batch_index=batch_index,
                    batch_size=len(batch),
                )

        if trace is not None:
            trace.record_stage(
                "batch_processor",
                elapsed_ms=(perf_counter() - started) * 1000,
                total_chunks=len(chunks),
                batch_size=self.batch_size,
                batch_count=len(list(self._iter_batches(chunks, self.batch_size))),
            )
        return all_records

    @staticmethod
    def _merge_records(dense_records: Sequence[ChunkRecord], sparse_records: Sequence[ChunkRecord]) -> List[ChunkRecord]:
        sparse_map = {record.id: record for record in sparse_records}
        merged: List[ChunkRecord] = []
        for dense in dense_records:
            sparse = sparse_map.get(dense.id)
            if sparse is None:
                raise ValueError(f"sparse record missing for chunk id={dense.id}")
            merged.append(
                ChunkRecord(
                    id=dense.id,
                    text=dense.text,
                    metadata=dict(dense.metadata),
                    dense_vector=dense.dense_vector,
                    sparse_vector=sparse.sparse_vector,
                )
            )
        return merged

    @staticmethod
    def _iter_batches(chunks: Sequence[Chunk], batch_size: int) -> Iterable[List[Chunk]]:
        for start in range(0, len(chunks), batch_size):
            yield list(chunks[start : start + batch_size])

    @staticmethod
    def _read_batch_size(settings: Any) -> int:
        if isinstance(settings, dict):
            ingestion = settings.get("ingestion", {})
            if isinstance(ingestion, dict):
                batch_cfg = ingestion.get("batch_processor", {})
                if isinstance(batch_cfg, dict):
                    return int(batch_cfg.get("batch_size", 16))
        if hasattr(settings, "ingestion") and hasattr(settings.ingestion, "batch_processor"):
            batch_cfg = settings.ingestion.batch_processor
            if hasattr(batch_cfg, "batch_size"):
                return int(batch_cfg.batch_size)
        return 16
