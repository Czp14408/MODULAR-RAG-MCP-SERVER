"""C10: BatchProcessor 批处理测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.batch_processor import BatchProcessor


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.batch_sizes: List[int] = []

    def encode(self, chunks: List[Chunk], trace=None) -> List[ChunkRecord]:
        self.batch_sizes.append(len(chunks))
        return [
            ChunkRecord(
                id=c.id,
                text=c.text,
                metadata=dict(c.metadata),
                dense_vector=[float(i + 1), float(i + 2)],
            )
            for i, c in enumerate(chunks)
        ]


class FakeSparseEncoder:
    def __init__(self) -> None:
        self.batch_sizes: List[int] = []

    def encode(self, chunks: List[Chunk], trace=None) -> List[ChunkRecord]:
        self.batch_sizes.append(len(chunks))
        return [
            ChunkRecord(
                id=c.id,
                text=c.text,
                metadata=dict(c.metadata),
                sparse_vector={"token": 1.0},
            )
            for c in chunks
        ]


def _chunks(n: int) -> List[Chunk]:
    return [
        Chunk(
            id=f"chunk-{i}",
            text=f"text-{i}",
            metadata={"source_path": "tests/data/test_chunking_text.pdf"},
            start_offset=0,
            end_offset=6,
            source_ref="doc",
        )
        for i in range(n)
    ]


def test_batch_processor_splits_into_expected_batches_and_preserves_order() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    # 参数选择说明：
    # n=5, batch_size=2 -> 期望 3 批（2/2/1），用于覆盖“尾批不足”场景。
    processor = BatchProcessor(settings={}, dense_encoder=dense, sparse_encoder=sparse, batch_size=2)
    records = processor.process(_chunks(5))

    print(f"[C10] dense_batch_sizes={dense.batch_sizes}")
    print(f"[C10] sparse_batch_sizes={sparse.batch_sizes}")
    print(f"[C10] record_ids={[r.id for r in records]}")

    assert dense.batch_sizes == [2, 2, 1]
    assert sparse.batch_sizes == [2, 2, 1]
    assert [r.id for r in records] == [f"chunk-{i}" for i in range(5)]
    assert all(r.dense_vector is not None for r in records)
    assert all(r.sparse_vector is not None for r in records)
