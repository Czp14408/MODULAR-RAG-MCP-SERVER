"""RagasEvaluator：封装 ragas 评估接口。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator, validate_eval_input


class RagasEvaluator(BaseEvaluator):
    """最小 ragas 封装。

    当前阶段为了保证项目可在未安装 ragas 时也能离线运行，
    真正调用前会先做依赖检查；单测通过 monkeypatch `_run_ragas`
    隔离三方框架。
    """

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        validate_eval_input(query, retrieved_ids, golden_ids)
        self._ensure_ragas_available()
        return self._run_ragas(query, retrieved_ids, golden_ids, trace=trace)

    @staticmethod
    def _ensure_ragas_available() -> None:
        try:
            import ragas  # type: ignore # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RagasEvaluator requires optional dependency `ragas`. "
                "Install it with `pip install ragas` before using provider=ragas."
            ) from exc

    def _run_ragas(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        """占位实现：实际框架接入后可替换为真实 ragas 调用。"""
        overlap = len(set(retrieved_ids) & set(golden_ids))
        precision = overlap / float(len(retrieved_ids) or 1)
        recall = overlap / float(len(golden_ids) or 1)
        return {
            "faithfulness": round(precision, 4),
            "answer_relevancy": round(recall, 4),
            "context_precision": round(precision, 4),
        }
