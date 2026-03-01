"""Reranker 工厂。"""

from typing import Any, Dict, Type

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.llm_reranker import LLMReranker


class RerankerFactoryError(ValueError):
    """Reranker 工厂错误。"""


class RerankerFactory:
    """根据配置创建重排序器实现。"""

    _providers: Dict[str, Type[BaseReranker]] = {
        "none": NoneReranker,
        "llm": LLMReranker,
        "cross_encoder": CrossEncoderReranker,
        "cross-encoder": CrossEncoderReranker,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseReranker]) -> None:
        """注册自定义重排序器（测试/扩展场景）。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, settings: Any) -> BaseReranker:
        """根据 rerank.provider 或 rerank.enabled 决定创建哪种实现。"""
        provider = cls._resolve_provider(settings)
        provider_cls = cls._providers.get(provider.lower())
        if provider_cls is None:
            supported = ", ".join(sorted(cls._providers))
            raise RerankerFactoryError(
                f"Unsupported rerank.provider: {provider}. Supported providers: {supported}"
            )
        return provider_cls(settings)

    @staticmethod
    def _resolve_provider(settings: Any) -> str:
        # dataclass Settings：若 only enabled 且为 False，则默认 none。
        if hasattr(settings, "rerank"):
            rerank = settings.rerank
            if hasattr(rerank, "provider") and getattr(rerank, "provider"):
                return str(getattr(rerank, "provider"))
            if hasattr(rerank, "enabled") and getattr(rerank, "enabled") is False:
                return "none"

        # dict 设置：兼容 {rerank: {provider: ...}} 与 {rerank: {enabled: false}}
        if isinstance(settings, dict):
            rerank = settings.get("rerank")
            if isinstance(rerank, dict):
                if "provider" in rerank and rerank["provider"]:
                    return str(rerank["provider"])
                if rerank.get("enabled") is False:
                    return "none"

        raise RerankerFactoryError("Missing or invalid field: rerank.provider")
