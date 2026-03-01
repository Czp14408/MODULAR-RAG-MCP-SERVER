"""Azure Embedding provider 的最小可运行桩实现。"""

from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class AzureEmbedding(BaseEmbedding):
    """用于阶段性验证工厂路由，不依赖真实外部 API。"""

    def embed(self, texts: List[str], trace: Optional[Any] = None) -> List[List[float]]:
        # 与其它桩保持同构，便于后续替换真实实现。
        return [[float(len(text)), float(sum(ord(ch) for ch in text) % 991)] for text in texts]
