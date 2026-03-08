"""H1: RagasEvaluator 测试。"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observability.evaluation.ragas_evaluator import RagasEvaluator


def test_ragas_evaluator_returns_metrics_with_mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = RagasEvaluator(settings={})
    monkeypatch.setattr(evaluator, "_ensure_ragas_available", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "_run_ragas",
        lambda query, retrieved_ids, golden_ids, trace=None: {  # noqa: ARG005
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "context_precision": 0.7,
        },
    )

    metrics = evaluator.evaluate("q", ["a"], ["a"])
    print(f"[H1] metrics={metrics}")

    assert metrics["faithfulness"] == 0.8
    assert metrics["answer_relevancy"] == 0.9


def test_ragas_evaluator_import_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = RagasEvaluator(settings={})

    def _boom() -> None:
        raise ImportError("missing ragas")

    monkeypatch.setattr(evaluator, "_ensure_ragas_available", _boom)
    with pytest.raises(ImportError, match="missing ragas"):
        evaluator.evaluate("q", ["a"], ["a"])
