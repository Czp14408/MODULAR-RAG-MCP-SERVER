"""Embedding provider 工厂。"""

from typing import Any, Dict, Type

from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding


class EmbeddingFactoryError(ValueError):
    """Embedding 工厂错误。"""


class EmbeddingFactory:
    """根据配置创建 Embedding 实现。"""

    _providers: Dict[str, Type[BaseEmbedding]] = {
        "openai": OpenAIEmbedding,
        "azure": AzureEmbedding,
        "ollama": OllamaEmbedding,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseEmbedding]) -> None:
        """注册自定义 provider（测试或扩展实现）。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, settings: Any) -> BaseEmbedding:
        """按 embedding.provider 构造实例。"""
        provider = cls._resolve_provider(settings)
        provider_cls = cls._providers.get(provider.lower())
        if provider_cls is None:
            supported = ", ".join(sorted(cls._providers))
            raise EmbeddingFactoryError(
                f"Unsupported embedding.provider: {provider}. Supported providers: {supported}"
            )
        return provider_cls(settings)

    @staticmethod
    def _resolve_provider(settings: Any) -> str:
        # 支持 dataclass Settings（主流程）和 dict（测试脚本）输入。
        if hasattr(settings, "embedding") and hasattr(settings.embedding, "provider"):
            return str(settings.embedding.provider)
        if isinstance(settings, dict):
            embedding_settings = settings.get("embedding")
            if isinstance(embedding_settings, dict) and "provider" in embedding_settings:
                return str(embedding_settings["provider"])
        raise EmbeddingFactoryError("Missing or invalid field: embedding.provider")
