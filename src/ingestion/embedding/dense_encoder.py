"""DenseEncoder：调用 libs.embedding 生成稠密向量。"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    """将 Chunk 文本批量编码为 dense vectors。"""

    def __init__(self, settings: Any, embedding: Optional[BaseEmbedding] = None) -> None:
        self.settings = settings
        self.embedding = embedding or EmbeddingFactory.create(settings)

    def encode(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[ChunkRecord]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding.embed(texts, trace=trace)
        if len(vectors) != len(chunks):
            raise ValueError("embedding output size mismatch with chunk count")

        records: List[ChunkRecord] = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    dense_vector=[float(v) for v in vector],
                )
            )

        if trace is not None:
            trace.record_stage(
                "dense_encoder",
                elapsed_ms=0.0,
                chunk_count=len(chunks),
                vector_dim=len(records[0].dense_vector or []),
            )
        return records
