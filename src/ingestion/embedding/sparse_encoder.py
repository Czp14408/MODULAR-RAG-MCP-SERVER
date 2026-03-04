"""SparseEncoder：构建 BM25 可用的稀疏权重表示。"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord, SparseVector


class SparseEncoder:
    """将 Chunk 文本编码为 term->weight 的稀疏向量。"""

    def encode(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[ChunkRecord]:
        records: List[ChunkRecord] = []
        for chunk in chunks:
            sparse = self._encode_text(chunk.text)
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    sparse_vector=sparse,
                )
            )

        if trace is not None:
            trace.record_stage(
                "sparse_encoder",
                elapsed_ms=0.0,
                chunk_count=len(chunks),
            )
        return records

    def _encode_text(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        counter = Counter(tokens)
        total = float(sum(counter.values()))
        # 参数选择说明：
        # 使用 TF 归一化作为当前阶段的最小可用权重定义，后续 C11 再叠加 IDF。
        return {term: (count / total) for term, count in counter.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        # 同时提取中英文词，避免中文文本被完全忽略。
        return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text.lower())
