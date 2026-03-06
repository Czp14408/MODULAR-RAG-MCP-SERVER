"""D3: SparseRetriever 测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeBM25Indexer(BM25Indexer):
    def __init__(self) -> None:
        super().__init__(persist_dir="data/db/test-bm25")
        self.calls: List[Dict[str, Any]] = []

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, object]]:
        self.calls.append({"query_text": query_text, "top_k": top_k})
        return [
            {"id": "chunk-a", "score": 1.2},
            {"id": "chunk-b", "score": 0.8},
        ]


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.get_calls: List[List[str]] = []

    def upsert(self, records, trace=None) -> None:
        return None

    def query(self, vector: List[float], top_k: int, filters=None, trace=None) -> List[Dict[str, Any]]:
        return []

    def get_by_ids(self, ids: List[str], trace=None) -> List[Dict[str, Any]]:
        self.get_calls.append(ids)
        return [
            {
                "id": "chunk-a",
                "text": "A 文本",
                "metadata": {"source_path": "tests/data/test_chunking_text.pdf", "lang": "zh"},
            },
            {
                "id": "chunk-b",
                "text": "B 文本",
                "metadata": {"source_path": "tests/data/test_chunking_text.pdf", "lang": "zh"},
            },
        ]


def test_sparse_retriever_merges_scores_with_text_and_metadata() -> None:
    bm25 = FakeBM25Indexer()
    store = FakeVectorStore(settings={})
    retriever = SparseRetriever(settings={}, bm25_indexer=bm25, vector_store=store)

    results = retriever.retrieve(["rag", "检索"], top_k=2)
    print(f"[D3] bm25_calls={bm25.calls}")
    print(f"[D3] get_by_ids_calls={store.get_calls}")
    print(f"[D3] results={[item.to_dict() for item in results]}")

    assert bm25.calls[0] == {"query_text": "rag 检索", "top_k": 2}
    assert store.get_calls[0] == ["chunk-a", "chunk-b"]
    assert results[0].chunk_id == "chunk-a"
    assert results[0].text == "A 文本"
    assert results[0].score == 1.2
