"""DocumentManager：跨存储的文档管理协调层。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore


class DocumentManager:
    """协调向量库、BM25、图片索引、完整性记录的文档生命周期。"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: SQLiteIntegrityChecker,
    ) -> None:
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        filters = {"collection": collection} if collection else {}
        rows = self.chroma_store.get_by_metadata(filters)
        grouped: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            metadata = dict(row.get("metadata", {}))
            doc_id = str(metadata.get("document_id") or metadata.get("source_path") or row["id"])
            item = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source_path": metadata.get("source_path", ""),
                    "collection": metadata.get("collection", ""),
                    "title": metadata.get("title", ""),
                    "summary": metadata.get("summary", ""),
                    "tags": metadata.get("tags", []),
                    "chunk_count": 0,
                    "image_count": 0,
                    "doc_hash": metadata.get("document_id", ""),
                },
            )
            item["chunk_count"] += 1
            images = metadata.get("images", [])
            if isinstance(images, list):
                item["image_count"] = max(item["image_count"], len(images))

        return sorted(grouped.values(), key=lambda item: (str(item["collection"]), str(item["source_path"])))

    def get_document_detail(self, doc_id: str) -> Dict[str, Any]:
        rows = self.chroma_store.get_by_metadata({"document_id": doc_id})
        if not rows:
            rows = [
                row
                for row in self.chroma_store.get_by_metadata({})
                if str(row.get("metadata", {}).get("source_path", "")).endswith(doc_id)
            ]
        if not rows:
            raise ValueError(f"document not found: {doc_id}")

        first = rows[0]
        metadata = dict(first.get("metadata", {}))
        doc_hash = str(metadata.get("document_id", ""))
        images = self.image_storage.list_images(
            collection=str(metadata.get("collection", "")) or None,
            doc_hash=doc_hash or None,
        )
        return {
            "doc_id": doc_id,
            "source_path": metadata.get("source_path", ""),
            "collection": metadata.get("collection", ""),
            "title": metadata.get("title", ""),
            "summary": metadata.get("summary", ""),
            "tags": metadata.get("tags", []),
            "chunks": rows,
            "images": images,
        }

    def delete_document(self, source_path: str, collection: Optional[str] = None) -> Dict[str, Any]:
        filters = {"source_path": source_path}
        if collection:
            filters["collection"] = collection
        rows = self.chroma_store.get_by_metadata(filters)
        doc_ids = {
            str(row.get("metadata", {}).get("document_id", ""))
            for row in rows
            if str(row.get("metadata", {}).get("document_id", "")).strip()
        }

        deleted_chunks = self.chroma_store.delete_by_metadata(filters)
        deleted_bm25 = self.bm25_indexer.remove_document(source_path=source_path)
        deleted_images = 0
        for doc_id in doc_ids:
            deleted_images += self.image_storage.delete_by_doc_hash(doc_id, collection=collection)
        deleted_integrity = self.file_integrity.remove_record(file_path=source_path)

        return {
            "source_path": source_path,
            "collection": collection,
            "deleted_chunks": deleted_chunks,
            "deleted_bm25": deleted_bm25,
            "deleted_images": deleted_images,
            "deleted_integrity": deleted_integrity,
        }

    def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        documents = self.list_documents(collection=collection)
        images = self.image_storage.list_images(collection=collection)
        vector_stats = self.chroma_store.get_collection_stats(collection=collection)
        return {
            "collection": collection or "all",
            "document_count": len(documents),
            "chunk_count": vector_stats.get("chunk_count", 0),
            "image_count": len(images),
            "collections": sorted(
                {
                    str(item.get("collection", ""))
                    for item in self.list_documents()
                    if str(item.get("collection", "")).strip()
                }
            ),
        }
