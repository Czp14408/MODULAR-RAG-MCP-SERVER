"""H2: CompositeEvaluator 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator


class _EvalA(BaseEvaluator):
    def evaluate(self, query, retrieved_ids, golden_ids, trace=None):  # noqa: ANN001, ANN201, ARG002
        return {"hit_rate": 1.0}


class _EvalB(BaseEvaluator):
    def evaluate(self, query, retrieved_ids, golden_ids, trace=None):  # noqa: ANN001, ANN201, ARG002
        return {"mrr": 0.5}


def test_composite_evaluator_merges_metrics() -> None:
    evaluator = CompositeEvaluator([_EvalA({}), _EvalB({})])
    metrics = evaluator.evaluate("q", ["a"], ["a"])
    print(f"[H2] metrics={metrics}")

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}
