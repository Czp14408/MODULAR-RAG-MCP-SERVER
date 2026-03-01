"""VectorStore 工厂。"""

from typing import Any, Dict, Type

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.chroma_store import ChromaStore


class VectorStoreFactoryError(ValueError):
    """VectorStore 工厂错误。"""


class VectorStoreFactory:
    """按配置创建向量存储实现。"""

    _providers: Dict[str, Type[BaseVectorStore]] = {
        "chroma": ChromaStore,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseVectorStore]) -> None:
        """注册自定义 provider（测试或扩展使用）。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, settings: Any) -> BaseVectorStore:
        """根据 vector_store.provider 实例化 provider。"""
        provider = cls._resolve_provider(settings)
        provider_cls = cls._providers.get(provider.lower())
        if provider_cls is None:
            supported = ", ".join(sorted(cls._providers))
            raise VectorStoreFactoryError(
                f"Unsupported vector_store.provider: {provider}. Supported providers: {supported}"
            )
        return provider_cls(settings)

    @staticmethod
    def _resolve_provider(settings: Any) -> str:
        # 支持 dataclass Settings 和 dict 两种输入。
        if hasattr(settings, "vector_store") and hasattr(settings.vector_store, "provider"):
            return str(settings.vector_store.provider)
        if isinstance(settings, dict):
            vs = settings.get("vector_store")
            if isinstance(vs, dict) and "provider" in vs:
                return str(vs["provider"])
        raise VectorStoreFactoryError("Missing or invalid field: vector_store.provider")
