"""DataService：封装文档与图片读取。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.ingestion.document_manager import DocumentManager


class DataService:
    """为 Dashboard 页面提供文档列表、详情和删除操作。"""

    def __init__(self, document_manager: DocumentManager) -> None:
        self.document_manager = document_manager

    def list_documents(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.document_manager.list_documents(collection=collection)

    def get_document_detail(self, doc_id: str) -> Dict[str, Any]:
        return self.document_manager.get_document_detail(doc_id)

    def delete_document(self, source_path: str, collection: Optional[str] = None) -> Dict[str, Any]:
        return self.document_manager.delete_document(source_path=source_path, collection=collection)

    def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        return self.document_manager.get_collection_stats(collection=collection)
