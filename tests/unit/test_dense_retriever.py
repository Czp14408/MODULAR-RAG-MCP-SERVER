"""D2: DenseRetriever 测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.dense_retriever import DenseRetriever
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeEmbedding(BaseEmbedding):
    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.calls: List[List[str]] = []

    def embed(self, texts: List[str], trace: Any = None) -> List[List[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3]]


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.query_calls: List[Dict[str, Any]] = []

    def upsert(self, records, trace=None) -> None:
        return None

    def query(self, vector: List[float], top_k: int, filters=None, trace=None) -> List[Dict[str, Any]]:
        self.query_calls.append({"vector": vector, "top_k": top_k, "filters": filters})
        return [
            {
                "id": "chunk-1",
                "score": 0.88,
                "text": "命中文本",
                "metadata": {"source_path": "tests/data/test_chunking_text.pdf"},
            }
        ]

    def get_by_ids(self, ids: List[str], trace=None) -> List[Dict[str, Any]]:
        return []


def test_dense_retriever_calls_embedding_and_vector_store() -> None:
    embedding = FakeEmbedding(settings={})
    store = FakeVectorStore(settings={})
    retriever = DenseRetriever(settings={}, embedding_client=embedding, vector_store=store)

    results = retriever.retrieve("rag query", top_k=3, filters={"collection": "demo"})
    print(f"[D2] embedding_calls={embedding.calls}")
    print(f"[D2] vector_calls={store.query_calls}")
    print(f"[D2] results={[item.to_dict() for item in results]}")

    assert embedding.calls == [["rag query"]]
    assert store.query_calls[0]["top_k"] == 3
    assert store.query_calls[0]["filters"] == {"collection": "demo"}
    assert results[0].chunk_id == "chunk-1"
    assert results[0].text == "命中文本"
