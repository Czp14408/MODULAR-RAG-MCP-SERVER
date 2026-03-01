"""Tests for LLM factory provider routing."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory, LLMFactoryError


class FakeLLM(BaseLLM):
    # 测试用假实现：不依赖真实模型，只用于验证工厂分流是否正确。
    def chat(self, messages):
        return "fake-response"


def make_settings(provider: str) -> Settings:
    # 构造最小可用 Settings，仅替换 llm.provider 作为测试变量。
    return Settings(
        llm=LLMSettings(provider=provider),
        embedding=EmbeddingSettings(provider="openai"),
        vector_store=VectorStoreSettings(provider="chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(enabled=False),
        evaluation=EvaluationSettings(enabled=False),
        observability=ObservabilitySettings(log_level="INFO"),
    )


def test_factory_routes_to_registered_fake_provider() -> None:
    # 验证：注册后的 provider 能被正确创建并返回 FakeLLM 实例。
    provider_name = "fake-test-provider"
    LLMFactory.register_provider(provider_name, FakeLLM)

    llm = LLMFactory.create(make_settings(provider_name))

    assert isinstance(llm, FakeLLM)
    assert llm.chat([Message(role="user", content="hello")]) == "fake-response"


def test_factory_supports_dict_style_settings() -> None:
    # 验证：create 支持 dict 风格配置输入（便于轻量测试）。
    provider_name = "fake-dict-provider"
    LLMFactory.register_provider(provider_name, FakeLLM)

    llm = LLMFactory.create({"llm": {"provider": provider_name}})

    assert isinstance(llm, FakeLLM)


def test_factory_raises_on_unknown_provider() -> None:
    # 验证：未知 provider 会抛出可读错误。
    with pytest.raises(LLMFactoryError, match="Unsupported llm.provider"):
        LLMFactory.create(make_settings("unknown-provider"))


def test_factory_raises_on_missing_provider() -> None:
    # 验证：缺失 llm.provider 时明确报错。
    with pytest.raises(LLMFactoryError, match="llm.provider"):
        LLMFactory.create({"llm": {}})
