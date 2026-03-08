"""B6: CustomEvaluator 与 EvaluatorFactory 测试。"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorContractError
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError


class FakeEvaluator(BaseEvaluator):
    # 测试桩：返回固定指标验证工厂分流。
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        return {"hit_rate": 0.5, "mrr": 0.5}



def test_custom_evaluator_metrics_are_stable() -> None:
    evaluator = CustomEvaluator(settings={})

    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["d1", "d2", "d3"],
        golden_ids=["d2", "d9"],
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5


def test_custom_evaluator_zero_when_no_hit() -> None:
    evaluator = CustomEvaluator(settings={})

    metrics = evaluator.evaluate(
        query="q",
        retrieved_ids=["a", "b"],
        golden_ids=["x"],
    )

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_factory_routes_to_registered_fake_provider() -> None:
    provider_name = "fake-evaluator"
    EvaluatorFactory.register_provider(provider_name, FakeEvaluator)

    evaluator = EvaluatorFactory.create({"evaluation": {"provider": provider_name}})

    assert isinstance(evaluator, FakeEvaluator)


def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(EvaluatorFactoryError, match="Unsupported evaluation.provider"):
        EvaluatorFactory.create({"evaluation": {"provider": "unknown"}})


def test_input_contract_validation() -> None:
    evaluator = CustomEvaluator(settings={})
    with pytest.raises(EvaluatorContractError, match="retrieved_ids"):
        evaluator.evaluate(query="q", retrieved_ids=[1], golden_ids=["a"])  # type: ignore[list-item]


def test_custom_evaluator_handles_empty_retrieved_ids() -> None:
    evaluator = CustomEvaluator(settings={})
    metrics = evaluator.evaluate(query="q", retrieved_ids=[], golden_ids=["a"])
    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}
