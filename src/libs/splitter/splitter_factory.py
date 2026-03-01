"""Splitter 工厂：按配置选择切分策略实现。"""

from typing import Any, Dict, Type

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.fixed_length_splitter import FixedLengthSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.semantic_splitter import SemanticSplitter


class SplitterFactoryError(ValueError):
    """Splitter 工厂异常。"""


class SplitterFactory:
    """切分器提供者注册与构造入口。"""

    _providers: Dict[str, Type[BaseSplitter]] = {
        "recursive": RecursiveSplitter,
        "semantic": SemanticSplitter,
        "fixed": FixedLengthSplitter,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseSplitter]) -> None:
        """注册自定义切分器实现（测试或扩展场景使用）。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, settings: Any) -> BaseSplitter:
        """根据 splitter.provider 创建切分器实例。"""
        provider = cls._resolve_provider(settings)
        provider_cls = cls._providers.get(provider.lower())
        if provider_cls is None:
            supported = ", ".join(sorted(cls._providers))
            raise SplitterFactoryError(
                f"Unsupported splitter.provider: {provider}. Supported providers: {supported}"
            )
        return provider_cls(settings)

    @staticmethod
    def _resolve_provider(settings: Any) -> str:
        # 支持 dataclass Settings 和 dict 两种输入形态。
        if hasattr(settings, "splitter") and hasattr(settings.splitter, "provider"):
            return str(settings.splitter.provider)
        if isinstance(settings, dict):
            splitter_settings = settings.get("splitter")
            if isinstance(splitter_settings, dict) and "provider" in splitter_settings:
                return str(splitter_settings["provider"])
        raise SplitterFactoryError("Missing or invalid field: splitter.provider")
