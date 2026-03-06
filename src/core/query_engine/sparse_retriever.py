"""SparseRetriever：BM25 查询 + 向量库补全文本元数据。"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class SparseRetriever:
    """使用 BM25 路径执行关键词召回。"""

    def __init__(
        self,
        settings: Any,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.settings = settings
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        keywords: List[str],
        top_k: int,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        query_text = " ".join(keywords).strip()
        ranked = self.bm25_indexer.query(query_text, top_k=top_k)
        rows = self.vector_store.get_by_ids([str(item["id"]) for item in ranked], trace=trace)
        row_map = {str(item["id"]): item for item in rows}

        results: List[RetrievalResult] = []
        for item in ranked:
            chunk_id = str(item["id"])
            payload = row_map.get(chunk_id, {})
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(item["score"]),
                    text=str(payload.get("text", item.get("text", ""))),
                    metadata=dict(payload.get("metadata", item.get("metadata", {}))),
                )
            )

        if trace is not None:
            trace.record_stage(
                "sparse_retrieval",
                elapsed_ms=0.0,
                top_k=top_k,
                result_count=len(results),
            )
        return results
