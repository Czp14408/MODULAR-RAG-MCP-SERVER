"""Evaluator 抽象契约。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EvaluatorContractError(ValueError):
    """评估器输入不满足契约时抛出的错误。"""


class BaseEvaluator(ABC):
    """统一评估接口。"""

    def __init__(self, settings: Any) -> None:
        # 保留设置对象，便于后续读取评估参数。
        self.settings = settings

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        """输出评估指标字典。"""


def validate_eval_input(query: str, retrieved_ids: List[str], golden_ids: List[str]) -> None:
    """校验评估输入 shape。"""
    if not isinstance(query, str):
        raise EvaluatorContractError("Invalid query: must be str")
    if not isinstance(retrieved_ids, list) or any(not isinstance(x, str) for x in retrieved_ids):
        raise EvaluatorContractError("Invalid retrieved_ids: must be list[str]")
    if not isinstance(golden_ids, list) or any(not isinstance(x, str) for x in golden_ids):
        raise EvaluatorContractError("Invalid golden_ids: must be list[str]")
