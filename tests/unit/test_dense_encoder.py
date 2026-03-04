"""C8: DenseEncoder 测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    def __init__(self, settings: Any, dim: int = 4, mismatch: bool = False) -> None:
        super().__init__(settings)
        self.dim = dim
        self.mismatch = mismatch

    def embed(self, texts: List[str], trace: Any = None) -> List[List[float]]:
        count = len(texts) - 1 if self.mismatch and texts else len(texts)
        return [[float(i + 1)] * self.dim for i in range(count)]


def _chunks(n: int) -> List[Chunk]:
    return [
        Chunk(
            id=f"c{i}",
            text=f"text-{i}",
            metadata={"source_path": "tests/data/test_chunking_text.pdf"},
            start_offset=0,
            end_offset=6,
            source_ref="doc",
        )
        for i in range(n)
    ]


def test_dense_encoder_outputs_same_count_and_dim() -> None:
    encoder = DenseEncoder(settings={}, embedding=FakeEmbedding(settings={}, dim=6))
    records = encoder.encode(_chunks(3))
    print(f"[C8] record_count={len(records)} dim={len(records[0].dense_vector or [])}")

    assert len(records) == 3
    assert all(len(r.dense_vector or []) == 6 for r in records)


def test_dense_encoder_raises_on_size_mismatch() -> None:
    encoder = DenseEncoder(settings={}, embedding=FakeEmbedding(settings={}, mismatch=True))
    with pytest.raises(ValueError, match="output size mismatch"):
        encoder.encode(_chunks(2))
