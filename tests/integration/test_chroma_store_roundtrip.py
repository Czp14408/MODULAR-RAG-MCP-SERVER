"""B7.6: ChromaStore upsert->query roundtrip 集成测试。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def test_chroma_store_roundtrip_with_persistence_and_filters(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma_store"

    settings = {
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(persist_dir),
        }
    }

    store = VectorStoreFactory.create(settings)

    store.upsert(
        [
            {
                "id": "doc-1",
                "vector": [1.0, 0.0],
                "metadata": {"collection": "a", "lang": "zh"},
                "text": "alpha",
            },
            {
                "id": "doc-2",
                "vector": [0.9, 0.1],
                "metadata": {"collection": "a", "lang": "en"},
                "text": "beta",
            },
            {
                "id": "doc-3",
                "vector": [0.0, 1.0],
                "metadata": {"collection": "b", "lang": "zh"},
                "text": "gamma",
            },
        ]
    )

    result = store.query([1.0, 0.0], top_k=2, filters={"collection": "a"})
    assert [item["id"] for item in result] == ["doc-1", "doc-2"]

    # 重新创建 store，验证持久化 roundtrip。
    store_reloaded = VectorStoreFactory.create(settings)
    result_reloaded = store_reloaded.query([1.0, 0.0], top_k=2, filters={"collection": "a"})
    assert [item["id"] for item in result_reloaded] == ["doc-1", "doc-2"]
