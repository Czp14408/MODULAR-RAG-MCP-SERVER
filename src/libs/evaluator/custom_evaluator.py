"""自定义轻量评估器：hit_rate + mrr。"""

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator, validate_eval_input


class CustomEvaluator(BaseEvaluator):
    """输出可回归的基础检索指标。"""

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        validate_eval_input(query, retrieved_ids, golden_ids)

        golden_set = set(golden_ids)
        hit_rate = 1.0 if any(doc_id in golden_set for doc_id in retrieved_ids) else 0.0

        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in golden_set:
                mrr = 1.0 / float(rank)
                break

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
        }
