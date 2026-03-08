"""HybridSearch：编排 query processing + dense + sparse + fusion。"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import Fusion
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class HybridSearch:
    """混合检索流程编排。"""

    def __init__(
        self,
        settings: Any,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[Fusion] = None,
    ) -> None:
        self.settings = settings
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever(settings)
        self.fusion = fusion or Fusion()
        self.last_debug: Dict[str, Any] = {}

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        query_started = perf_counter()
        processed = self.query_processor.process(query, filters=filters)
        merged_filters = dict(processed.filters)
        if trace is not None:
            trace.record_stage(
                "query_processing",
                elapsed_ms=(perf_counter() - query_started) * 1000,
                method="keyword_extraction",
                provider=self.query_processor.__class__.__name__,
                keyword_count=len(processed.keywords),
                filters=dict(merged_filters),
            )

        dense_results: List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None

        try:
            dense_results = self.dense_retriever.retrieve(
                query=processed.query,
                top_k=top_k,
                filters=merged_filters,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            dense_error = str(exc)

        try:
            sparse_results = self.sparse_retriever.retrieve(
                keywords=processed.keywords,
                top_k=top_k,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            sparse_error = str(exc)

        fusion_started = perf_counter()
        if dense_results and sparse_results:
            fused = self.fusion.fuse(dense_results, sparse_results, top_k=top_k)
        elif dense_results:
            fused = dense_results[:top_k]
        else:
            fused = sparse_results[:top_k]

        filtered = self._apply_metadata_filters(fused, merged_filters)[:top_k]
        self.last_debug = {
            "processed_query": processed,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fusion_results": filtered,
            "dense_error": dense_error,
            "sparse_error": sparse_error,
        }

        if trace is not None:
            trace.record_stage(
                "fusion",
                elapsed_ms=(perf_counter() - fusion_started) * 1000,
                dense_count=len(dense_results),
                sparse_count=len(sparse_results),
                fused_count=len(filtered),
                method="rrf",
                provider=self.fusion.__class__.__name__,
            )
        return filtered

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        if not filters:
            return candidates
        output: List[RetrievalResult] = []
        for item in candidates:
            if all(item.metadata.get(key) == value for key, value in filters.items()):
                output.append(item)
        return output
