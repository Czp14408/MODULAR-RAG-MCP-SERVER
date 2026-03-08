"""EvalRunner：读取 golden set 并执行检索评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.evaluator.base_evaluator import BaseEvaluator


class EvalRunner:
    """执行测试集并汇总评估报告。"""

    def __init__(
        self,
        settings: Any,
        hybrid_search: HybridSearch,
        evaluator: BaseEvaluator,
    ) -> None:
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator

    def run(self, test_set_path: str) -> Dict[str, Any]:
        data = json.loads(Path(test_set_path).read_text(encoding="utf-8"))
        cases = list(data.get("test_cases", []))
        results: List[Dict[str, Any]] = []

        for case in cases:
            query = str(case.get("query", ""))
            top_k = int(case.get("top_k", self._default_top_k()))
            filters = case.get("filters")
            retrieval_results = self.hybrid_search.search(query=query, top_k=top_k, filters=filters)
            retrieved_chunk_ids = [item.chunk_id for item in retrieval_results]
            retrieved_sources = [
                Path(str(item.metadata.get("source_path", ""))).name for item in retrieval_results
            ]

            golden_chunk_ids = list(case.get("expected_chunk_ids", []) or [])
            golden_sources = list(case.get("expected_sources", []) or [])
            eval_retrieved = retrieved_chunk_ids if golden_chunk_ids else retrieved_sources
            eval_golden = golden_chunk_ids if golden_chunk_ids else golden_sources

            metrics = self.evaluator.evaluate(query, eval_retrieved, eval_golden)
            results.append(
                {
                    "query": query,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_sources": retrieved_sources,
                    "metrics": metrics,
                }
            )

        return {
            "case_count": len(results),
            "metrics": self._aggregate_metrics(results),
            "results": results,
        }

    def _default_top_k(self) -> int:
        if hasattr(self.settings, "retrieval") and hasattr(self.settings.retrieval, "top_k"):
            return int(self.settings.retrieval.top_k)
        if isinstance(self.settings, dict):
            retrieval = self.settings.get("retrieval", {})
            if isinstance(retrieval, dict):
                return int(retrieval.get("top_k", 5))
        return 5

    @staticmethod
    def _aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        metric_names = {name for item in results for name in item["metrics"].keys()}
        aggregated: Dict[str, float] = {}
        for name in metric_names:
            aggregated[name] = round(
                sum(float(item["metrics"].get(name, 0.0)) for item in results) / float(len(results)),
                4,
            )
        return aggregated
