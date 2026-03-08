"""DenseRetriever：query embedding + vector store 检索。"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetriever:
    """使用 dense vector 执行语义召回。"""

    def __init__(
        self,
        settings: Any,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingFactory.create(settings)
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        started = perf_counter()
        vector = self.embedding_client.embed([query], trace=trace)[0]
        rows = self.vector_store.query(vector, top_k=top_k, filters=filters, trace=trace)
        results = [
            RetrievalResult(
                chunk_id=str(item["id"]),
                score=float(item["score"]),
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata", {})),
            )
            for item in rows
        ]

        if trace is not None:
            trace.record_stage(
                "dense_retrieval",
                elapsed_ms=(perf_counter() - started) * 1000,
                top_k=top_k,
                result_count=len(results),
                method="dense_vector_search",
                provider=self.embedding_client.__class__.__name__,
            )
        return results
