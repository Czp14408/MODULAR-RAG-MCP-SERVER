"""D6: Core QueryReranker fallback 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.reranker import QueryReranker
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, RerankerFallbackSignal


class FailingReranker(BaseReranker):
    def rerank(self, query, candidates, trace=None):  # noqa: ANN001
        raise RerankerFallbackSignal("provider=test error_type=Mock detail=boom")


class StableReranker(BaseReranker):
    def rerank(self, query, candidates, trace=None):  # noqa: ANN001
        return list(reversed(candidates))


def _candidate(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text-{chunk_id}",
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
    )


def test_query_reranker_fallbacks_without_breaking_results() -> None:
    reranker = QueryReranker(settings={}, reranker=FailingReranker(settings={}))
    candidates = [_candidate("a", 0.9), _candidate("b", 0.8)]

    results = reranker.rerank("query", candidates)
    print(f"[D6] fallback_results={[item.to_dict() for item in results]}")
    print(f"[D6] fallback_state={reranker.last_fallback} error={reranker.last_error}")

    assert reranker.last_fallback is True
    assert results[0].metadata["rerank_fallback"] is True
    assert [item.chunk_id for item in results] == ["a", "b"]


def test_query_reranker_uses_backend_result_when_successful() -> None:
    reranker = QueryReranker(settings={}, reranker=StableReranker(settings={}))
    candidates = [_candidate("a", 0.9), _candidate("b", 0.8)]

    results = reranker.rerank("query", candidates)
    print(f"[D6] reranked_results={[item.to_dict() for item in results]}")

    assert reranker.last_fallback is False
    assert [item.chunk_id for item in results] == ["b", "a"]
