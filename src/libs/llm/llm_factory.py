"""Factory for creating LLM providers by configuration."""

from typing import Any, Dict, Type

from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.openai_llm import OpenAILLM


class LLMFactoryError(ValueError):
    """LLM 工厂创建失败时抛出的异常。"""


class LLMFactory:
    """LLM 后端提供者的注册表与构造入口。"""

    _providers: Dict[str, Type[BaseLLM]] = {
        # 这里维护 provider 名称到实现类的映射，后续可按阶段继续扩展。
        "openai": OpenAILLM,
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "ollama": OllamaLLM,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseLLM]) -> None:
        """注册自定义 provider（测试 stub 或新后端都走这里）。"""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, settings: Any) -> BaseLLM:
        """根据配置解析 provider 并实例化对应实现类。"""
        provider = cls._resolve_provider(settings)
        provider_cls = cls._providers.get(provider.lower())
        if provider_cls is None:
            supported = ", ".join(sorted(cls._providers))
            raise LLMFactoryError(
                f"Unsupported llm.provider: {provider}. Supported providers: {supported}"
            )
        return provider_cls(settings)

    @staticmethod
    def _resolve_provider(settings: Any) -> str:
        # 支持 dataclass Settings（主流程）和 dict（测试/脚本）两种输入形态。
        if hasattr(settings, "llm") and hasattr(settings.llm, "provider"):
            return str(settings.llm.provider)
        if isinstance(settings, dict):
            llm_settings = settings.get("llm")
            if isinstance(llm_settings, dict) and "provider" in llm_settings:
                return str(llm_settings["provider"])
        raise LLMFactoryError("Missing or invalid field: llm.provider")
