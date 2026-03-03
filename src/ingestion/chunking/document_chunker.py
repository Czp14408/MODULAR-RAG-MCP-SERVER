"""DocumentChunker：Document -> Chunk 业务适配层（C4）。"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Dict, List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """将 libs.splitter 的纯文本切分结果，转换为 core.types 的 Chunk 对象。"""

    def __init__(self, settings: Any) -> None:
        # 参数选择说明：
        # 1) 接收 settings 的类型保持宽松（dataclass/dict 都支持），便于测试与后续演进。
        # 2) 若外部配置缺失 splitter.provider，则默认回退到 recursive，保证 C4 开箱即用。
        self.settings = settings
        self.splitter = SplitterFactory.create(self._build_effective_settings(settings))

    def split_document(self, document: Document) -> List[Chunk]:
        """执行完整的 Document -> Chunk 转换流程。"""
        pieces = self.splitter.split_text(document.text)
        chunks: List[Chunk] = []
        cursor = 0

        for index, piece in enumerate(pieces):
            # 为减少噪声，过滤纯空白块，避免污染索引。
            if not piece or not piece.strip():
                continue

            # 尽量定位块在原文中的真实偏移，若因归一化导致找不到则退化为当前 cursor。
            start_offset = document.text.find(piece, cursor)
            if start_offset < 0:
                start_offset = cursor
            end_offset = start_offset + len(piece)
            cursor = end_offset

            chunk = Chunk(
                id=self._generate_chunk_id(document.id, index, piece),
                text=piece,
                metadata=self._inherit_metadata(document=document, chunk_index=index),
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id,
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """生成稳定且可读的 Chunk ID：{doc_id}_{index:04d}_{hash8}。"""
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]  # noqa: S324
        return f"{doc_id}_{index:04d}_{digest}"

    def _inherit_metadata(self, document: Document, chunk_index: int) -> Dict[str, Any]:
        """继承文档元数据并注入 chunk_index。"""
        metadata = deepcopy(document.metadata)
        metadata["chunk_index"] = chunk_index
        return metadata

    def _build_effective_settings(self, settings: Any) -> Dict[str, Any]:
        """统一构造 splitter 工厂所需的配置字典。"""
        if isinstance(settings, dict):
            splitter_cfg = settings.get("splitter", {})
            if isinstance(splitter_cfg, dict) and "provider" in splitter_cfg:
                return settings

        provider = "recursive"
        chunk_size = 500
        chunk_overlap = 50
        if hasattr(settings, "splitter"):
            splitter = settings.splitter
            provider = str(getattr(splitter, "provider", provider))
            chunk_size = int(getattr(splitter, "chunk_size", chunk_size))
            chunk_overlap = int(getattr(splitter, "chunk_overlap", chunk_overlap))

        return {
            "splitter": {
                "provider": provider,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        }
