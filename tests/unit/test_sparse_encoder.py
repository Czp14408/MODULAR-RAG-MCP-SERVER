"""C9: SparseEncoder 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk
from src.ingestion.embedding.sparse_encoder import SparseEncoder


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(
        id=f"c{idx}",
        text=text,
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc",
    )


def test_sparse_encoder_outputs_term_weights_contract() -> None:
    encoder = SparseEncoder()
    records = encoder.encode([_chunk("RAG 检索 检索", 1)])
    sparse = records[0].sparse_vector or {}
    print(f"[C9] sparse={sparse}")

    assert "rag" in sparse
    assert "检索" in sparse
    assert abs(sum(sparse.values()) - 1.0) < 1e-9


def test_sparse_encoder_handles_empty_text_with_empty_vector() -> None:
    encoder = SparseEncoder()
    records = encoder.encode([_chunk("   ", 2)])
    print(f"[C9] empty sparse={records[0].sparse_vector}")

    assert records[0].sparse_vector == {}
