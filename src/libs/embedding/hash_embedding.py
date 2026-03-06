"""HashEmbedding：本地离线可用的确定性 embedding。"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class HashEmbedding(BaseEmbedding):
    """使用哈希值生成稳定向量，适合作为本地测试/离线默认实现。"""

    def embed(self, texts: List[str], trace: Optional[Any] = None) -> List[List[float]]:
        vectors: List[List[float]] = []
        dim = int(_read_option(self.settings, "dimension", 8))
        dim = max(1, dim)

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("HashEmbedding requires non-empty text")
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # 参数选择说明：
            # 取 sha256 前 dim 个字节并线性归一化到 [0, 1]，保证：
            # 1) 相同文本向量稳定；
            # 2) 不依赖外部模型或网络；
            # 3) 维度足够小，便于测试快速执行。
            vector = [round(byte / 255.0, 6) for byte in digest[:dim]]
            vectors.append(vector)

        return vectors


def _read_option(settings: Any, key: str, default: Any) -> Any:
    if hasattr(settings, "embedding") and hasattr(settings.embedding, key):
        value = getattr(settings.embedding, key)
        return default if value is None else value
    if isinstance(settings, dict):
        embedding = settings.get("embedding")
        if isinstance(embedding, dict):
            value = embedding.get(key, default)
            return default if value is None else value
    return default
