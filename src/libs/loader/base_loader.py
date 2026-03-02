"""Loader 抽象基类（C3）。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import Document


class BaseLoader(ABC):
    """所有文件 Loader 的统一抽象接口。"""

    @abstractmethod
    def load(self, path: str) -> Document:
        """加载单个文件并输出统一的 Document 契约对象。"""
