"""Ragas evaluator 的 libs 层导出。

这个文件保留在 libs 命名空间下，目的是让外部调用方在按
`src.libs.evaluator.ragas_evaluator` 路径查找实现时，不会拿到一个
"空壳占位文件"。真实实现仍放在 observability 层，因为它本质上属于
评估/观测能力，而不是核心检索链路。
"""

from src.observability.evaluation.ragas_evaluator import RagasEvaluator

__all__ = ["RagasEvaluator"]
