"""B2: EmbeddingFactory 工厂路由测试。"""

from pathlib import Path
import sys
from typing import Any, List, Optional

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
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError


class FakeEmbedding(BaseEmbedding):
    # 测试用假实现：返回稳定向量，不依赖外部服务。
    def embed(self, texts: List[str], trace: Optional[Any] = None) -> List[List[float]]:
        return [[42.0, float(i)] for i, _ in enumerate(texts)]


def make_settings(provider: str) -> Settings:
    # 构造最小可用 Settings，仅替换 embedding.provider。
    return Settings(
        llm=LLMSettings(provider="openai"),
        embedding=EmbeddingSettings(provider=provider),
        vector_store=VectorStoreSettings(provider="chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(enabled=False),
        evaluation=EvaluationSettings(enabled=False),
        observability=ObservabilitySettings(log_level="INFO"),
    )


def test_factory_routes_to_registered_fake_provider() -> None:
    provider_name = "fake-embedding-provider"
    EmbeddingFactory.register_provider(provider_name, FakeEmbedding)

    embedding = EmbeddingFactory.create(make_settings(provider_name))

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.embed(["a", "b"]) == [[42.0, 0.0], [42.0, 1.0]]


def test_factory_supports_dict_style_settings() -> None:
    provider_name = "fake-dict-embedding"
    EmbeddingFactory.register_provider(provider_name, FakeEmbedding)

    embedding = EmbeddingFactory.create({"embedding": {"provider": provider_name}})

    assert isinstance(embedding, FakeEmbedding)


def test_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(EmbeddingFactoryError, match="Unsupported embedding.provider"):
        EmbeddingFactory.create(make_settings("unknown-provider"))


def test_factory_raises_on_missing_provider() -> None:
    with pytest.raises(EmbeddingFactoryError, match="embedding.provider"):
        EmbeddingFactory.create({"embedding": {}})
