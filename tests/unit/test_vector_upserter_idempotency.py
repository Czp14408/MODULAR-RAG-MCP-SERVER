"""C12: VectorUpserter 幂等性测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import ChunkRecord
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.records: List[Dict[str, Any]] = []

    def upsert(self, records: Iterable[Dict[str, Any]], trace=None) -> None:
        self.records = list(records)

    def query(self, vector: List[float], top_k: int, filters=None, trace=None) -> List[Dict[str, Any]]:
        return []

    def get_by_ids(self, ids: List[str], trace=None) -> List[Dict[str, Any]]:
        return []

    def get_by_metadata(self, filters: Dict[str, Any], trace=None) -> List[Dict[str, Any]]:
        return []

    def delete_by_metadata(self, filters: Dict[str, Any], trace=None) -> int:
        return 0

    def get_collection_stats(self, collection=None, trace=None) -> Dict[str, Any]:
        return {"collection": collection, "count": len(self.records)}


def _record(text: str, chunk_index: int) -> ChunkRecord:
    return ChunkRecord(
        id="orig-id",
        text=text,
        metadata={"source_path": "tests/data/test_chunking_text.pdf", "chunk_index": chunk_index},
        dense_vector=[0.1, 0.2, 0.3],
    )


def test_same_chunk_two_upserts_produce_same_id() -> None:
    store = FakeVectorStore(settings={})
    upserter = VectorUpserter(settings={}, vector_store=store)

    first = upserter.upsert([_record("same text", 0)])[0]
    second = upserter.upsert([_record("same text", 0)])[0]
    print(f"[C12] first_id={first.id} second_id={second.id}")

    assert first.id == second.id


def test_content_change_produces_new_id() -> None:
    upserter = VectorUpserter(settings={}, vector_store=FakeVectorStore(settings={}))
    first = upserter.upsert([_record("text a", 0)])[0]
    second = upserter.upsert([_record("text b", 0)])[0]
    print(f"[C12] changed_ids={first.id}, {second.id}")

    assert first.id != second.id


def test_batch_upsert_preserves_order() -> None:
    store = FakeVectorStore(settings={})
    upserter = VectorUpserter(settings={}, vector_store=store)
    records = upserter.upsert([_record("a", 0), _record("b", 1), _record("c", 2)])
    print(f"[C12] ordered_ids={[r.id for r in records]}")

    assert len(records) == 3
    assert [item["text"] for item in store.records] == ["a", "b", "c"]
