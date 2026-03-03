"""Transform 抽象接口（C5）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk


class BaseTransform(ABC):
    """所有 Transform 组件的统一抽象。"""

    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """输入 Chunk 列表并输出转换后的 Chunk 列表。"""
