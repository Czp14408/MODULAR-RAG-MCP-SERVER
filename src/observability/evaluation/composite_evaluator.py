"""CompositeEvaluator：组合多个评估器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator, validate_eval_input


class CompositeEvaluator(BaseEvaluator):
    """顺序执行多个 evaluator 并合并 metrics。"""

    def __init__(self, evaluators: List[BaseEvaluator], settings: Any = None) -> None:
        super().__init__(settings or {})
        self.evaluators = evaluators

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        validate_eval_input(query, retrieved_ids, golden_ids)
        merged: Dict[str, float] = {}
        for evaluator in self.evaluators:
            merged.update(evaluator.evaluate(query, retrieved_ids, golden_ids, trace=trace))
        return merged
