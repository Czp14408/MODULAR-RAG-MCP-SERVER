"""G2: DocumentManager 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.document_manager import DocumentManager
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore


def _build_manager(tmp_path: Path) -> DocumentManager:
    store = ChromaStore(
        {
            "vector_store": {
                "provider": "chroma",
                "persist_directory": str(tmp_path / "data" / "db" / "chroma"),
            }
        }
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0],
                "text": "alpha",
                "metadata": {
                    "source_path": "docs/a.pdf",
                    "collection": "demo",
                    "document_id": "doc-a",
                    "title": "A",
                    "summary": "sum-a",
                    "tags": ["a"],
                },
            },
            {
                "id": "chunk-2",
                "vector": [0.0, 1.0],
                "text": "beta",
                "metadata": {
                    "source_path": "docs/a.pdf",
                    "collection": "demo",
                    "document_id": "doc-a",
                    "title": "A",
                    "summary": "sum-a",
                    "tags": ["a"],
                },
            },
        ]
    )
    bm25 = BM25Indexer(str(tmp_path / "data" / "db" / "bm25"))
    image_storage = ImageStorage(
        images_root=str(tmp_path / "data" / "images"),
        db_path=str(tmp_path / "data" / "db" / "image_index.db"),
    )
    image_storage.save_image("img-1", b"123", "demo", "doc-a")
    integrity = SQLiteIntegrityChecker(str(tmp_path / "data" / "db" / "ingestion_history.db"))
    integrity.mark_success("hash-a", "docs/a.pdf", file_size=1, chunk_count=2)
    return DocumentManager(store, bm25, image_storage, integrity)


def test_document_manager_lists_and_deletes_documents(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)

    docs = manager.list_documents(collection="demo")
    print(f"[G2] docs={docs}")
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == 2

    detail = manager.get_document_detail("doc-a")
    assert detail["title"] == "A"
    assert len(detail["chunks"]) == 2

    result = manager.delete_document("docs/a.pdf", collection="demo")
    print(f"[G2] delete_result={result}")
    assert result["deleted_chunks"] == 2
    assert manager.list_documents(collection="demo") == []
