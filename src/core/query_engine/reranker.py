"""Core 层 Reranker 编排与 fallback。"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import RerankerFallbackSignal
from src.libs.reranker.reranker_factory import RerankerFactory


class QueryReranker:
    """对 HybridSearch 候选做可回退重排。"""

    def __init__(self, settings: Any, reranker: Optional[Any] = None) -> None:
        self.settings = settings
        self.reranker = reranker or RerankerFactory.create(settings)
        self.last_fallback = False
        self.last_error: Optional[str] = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        self.last_fallback = False
        self.last_error = None
        if not candidates:
            return []

        payload = [
            {
                "id": item.chunk_id,
                "score": item.score,
                "text": item.text,
                "metadata": dict(item.metadata),
            }
            for item in candidates
        ]

        try:
            ranked = self.reranker.rerank(query=query, candidates=payload, trace=trace)
            results = [
                RetrievalResult(
                    chunk_id=str(item["id"]),
                    score=float(item.get("score", 0.0)),
                    text=str(item.get("text", "")),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in ranked
            ]
        except RerankerFallbackSignal as exc:
            self.last_fallback = True
            self.last_error = str(exc)
            results = [
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    score=item.score,
                    text=item.text,
                    metadata={**dict(item.metadata), "rerank_fallback": True},
                )
                for item in candidates
            ]
        except Exception as exc:  # noqa: BLE001
            self.last_fallback = True
            self.last_error = str(exc)
            results = [
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    score=item.score,
                    text=item.text,
                    metadata={**dict(item.metadata), "rerank_fallback": True},
                )
                for item in candidates
            ]

        if trace is not None:
            trace.record_stage(
                "rerank",
                elapsed_ms=0.0,
                candidate_count=len(candidates),
                fallback=self.last_fallback,
            )
        return results
