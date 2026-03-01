"""Reranker 抽象契约。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RerankerContractError(ValueError):
    """重排序输入或输出不满足契约时抛出的错误。"""


class RerankerFallbackSignal(RuntimeError):
    """重排序失败但可回退的信号异常。

    Core 层在接入时可捕获该异常并回退到融合排序结果。
    """


class BaseReranker(ABC):
    """统一重排序接口。"""

    def __init__(self, settings: Any) -> None:
        # 保留配置对象，便于子类读取 provider 参数。
        self.settings = settings

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """对候选列表进行重排序并返回新列表。"""


class NoneReranker(BaseReranker):
    """默认回退实现：保持输入顺序不变。"""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        _validate_candidates(candidates)
        return list(candidates)


def _validate_candidates(candidates: List[Dict[str, Any]]) -> None:
    """校验候选 shape，至少需要 list[dict]。"""
    if not isinstance(candidates, list):
        raise RerankerContractError("Invalid candidates: must be list[dict]")
    for item in candidates:
        if not isinstance(item, dict):
            raise RerankerContractError("Invalid candidate item: must be dict")
