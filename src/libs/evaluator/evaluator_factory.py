"""Evaluator 工厂。"""

from importlib import import_module
from typing import Any, Dict, Type, Union

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactoryError(ValueError):
    """Evaluator 工厂错误。"""


class EvaluatorFactory:
    """根据配置创建评估器。"""

    _providers: Dict[str, Union[Type[BaseEvaluator], str]] = {
        "custom": CustomEvaluator,
        "ragas": "src.observability.evaluation.ragas_evaluator:RagasEvaluator",
        "composite": "src.observability.evaluation.composite_evaluator:CompositeEvaluator",
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseEvaluator]) -> None:
        """注册自定义评估器（测试或扩展使用）。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, settings: Any) -> BaseEvaluator:
        """根据 evaluation.provider 创建评估器。"""
        provider = cls._resolve_provider(settings)
        provider_cls = cls._providers.get(provider.lower())
        if provider_cls is None:
            supported = ", ".join(sorted(cls._providers))
            raise EvaluatorFactoryError(
                f"Unsupported evaluation.provider: {provider}. Supported providers: {supported}"
            )
        if isinstance(provider_cls, str):
            module_name, class_name = provider_cls.split(":", 1)
            provider_cls = getattr(import_module(module_name), class_name)
            cls._providers[provider.lower()] = provider_cls
        return provider_cls(settings)

    @staticmethod
    def _resolve_provider(settings: Any) -> str:
        # dataclass Settings：当前只有 enabled 字段时，默认回退 custom。
        if hasattr(settings, "evaluation"):
            evaluation = settings.evaluation
            if hasattr(evaluation, "provider") and getattr(evaluation, "provider"):
                return str(getattr(evaluation, "provider"))
            if hasattr(evaluation, "enabled"):
                return "custom"

        # dict 设置：支持显式 provider，否则默认 custom。
        if isinstance(settings, dict):
            evaluation = settings.get("evaluation")
            if isinstance(evaluation, dict):
                if "provider" in evaluation and evaluation["provider"]:
                    return str(evaluation["provider"])
                return "custom"

        raise EvaluatorFactoryError("Missing or invalid field: evaluation.provider")
