"""D4: Fusion RRF 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.fusion import Fusion
from src.core.types import RetrievalResult


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text-{chunk_id}",
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
    )


def test_rrf_fusion_is_deterministic_and_configurable() -> None:
    fusion = Fusion(k=10)
    dense = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
    sparse = [_result("b", 1.2), _result("a", 1.0), _result("d", 0.5)]

    fused_1 = fusion.fuse(dense, sparse, top_k=3)
    fused_2 = fusion.fuse(dense, sparse, top_k=3)
    print(f"[D4] fused_1={[item.to_dict() for item in fused_1]}")
    print(f"[D4] fused_2={[item.to_dict() for item in fused_2]}")

    assert [item.chunk_id for item in fused_1] == [item.chunk_id for item in fused_2]
    assert len(fused_1) == 3
    assert fused_1[0].chunk_id in {"a", "b"}
