"""Embedding 提供者抽象接口。"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseEmbedding(ABC):
    """Embedding 后端的统一抽象。"""

    def __init__(self, settings: Any) -> None:
        # 保留设置对象，便于具体 provider 读取自己的参数。
        self.settings = settings

    @abstractmethod
    def embed(self, texts: List[str], trace: Optional[Any] = None) -> List[List[float]]:
        """批量将文本编码为向量。"""
