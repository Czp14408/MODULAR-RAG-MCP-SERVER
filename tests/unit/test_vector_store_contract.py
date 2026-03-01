"""B4: VectorStore 契约测试（输入输出 shape）。"""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.vector_store.base_vector_store import VectorStoreContractError
from src.libs.vector_store.vector_store_factory import VectorStoreFactory, VectorStoreFactoryError


def test_upsert_and_query_shape_contract() -> None:
    store = VectorStoreFactory.create({"vector_store": {"provider": "chroma"}})

    records = [
        {
            "id": "chunk-1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": {"collection": "a", "lang": "zh"},
            "text": "alpha",
        },
        {
            "id": "chunk-2",
            "vector": [0.9, 0.1, 0.0],
            "metadata": {"collection": "a", "lang": "en"},
            "text": "beta",
        },
    ]
    store.upsert(records)

    result = store.query([1.0, 0.0, 0.0], top_k=2, filters={"collection": "a"})

    assert isinstance(result, list)
    assert len(result) <= 2
    assert result
    for item in result:
        assert set(item.keys()) == {"id", "score", "metadata", "text"}
        assert isinstance(item["id"], str)
        assert isinstance(item["score"], float)
        assert isinstance(item["metadata"], dict)
        assert isinstance(item["text"], str)


def test_upsert_rejects_invalid_record_shape() -> None:
    store = VectorStoreFactory.create({"vector_store": {"provider": "chroma"}})

    with pytest.raises(VectorStoreContractError, match="record.id"):
        store.upsert([{"vector": [1.0, 2.0]}])



def test_query_rejects_invalid_vector_and_top_k() -> None:
    store = VectorStoreFactory.create({"vector_store": {"provider": "chroma"}})

    with pytest.raises(VectorStoreContractError, match="vector"):
        store.query([], top_k=1)

    with pytest.raises(VectorStoreContractError, match="top_k"):
        store.query([1.0, 2.0], top_k=0)



def test_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(VectorStoreFactoryError, match="Unsupported vector_store.provider"):
        VectorStoreFactory.create({"vector_store": {"provider": "unknown"}})
