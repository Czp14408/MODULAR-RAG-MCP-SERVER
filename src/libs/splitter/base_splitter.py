"""文本切分器抽象接口。"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseSplitter(ABC):
    """统一的切分器抽象，屏蔽不同策略实现差异。"""

    def __init__(self, settings: Any) -> None:
        # 保留设置对象，便于子类读取参数。
        self.settings = settings

    @abstractmethod
    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        """将输入文本切分为多个片段。"""
