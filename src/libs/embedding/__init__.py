"""Embedding 抽象与实现导出。"""

from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError
from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError
from src.libs.embedding.openai_embedding import (
    OpenAICompatibleEmbedding,
    OpenAIEmbedding,
    OpenAIEmbeddingError,
)

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "EmbeddingFactoryError",
    "OpenAICompatibleEmbedding",
    "OpenAIEmbedding",
    "OpenAIEmbeddingError",
    "AzureEmbedding",
    "OllamaEmbedding",
    "OllamaEmbeddingError",
]
