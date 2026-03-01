"""Azure OpenAI Embedding provider。"""

from __future__ import annotations

import json
from typing import Any
from urllib import request as urlrequest

from src.libs.embedding.openai_embedding import (
    OpenAICompatibleEmbedding,
    OpenAIEmbeddingError,
    _read_embedding_option,
)


class AzureEmbedding(OpenAICompatibleEmbedding):
    """Azure Embedding 实现。

    与 OpenAI 官方的主要差异：
    1. endpoint 包含 deployment 路径。
    2. 鉴权使用 `api-key` 头。
    3. 请求需附带 `api-version` 查询参数。
    """

    provider_name = "azure"

    def _build_embedding_endpoint(self) -> str:
        endpoint = str(_read_embedding_option(self.settings, "endpoint", default="")).rstrip("/")
        deployment = str(_read_embedding_option(self.settings, "deployment", default=""))
        api_version = str(_read_embedding_option(self.settings, "api_version", default="2024-02-01"))

        if not endpoint or not deployment:
            raise OpenAIEmbeddingError(
                "provider=azure error_type=ConfigError detail=missing endpoint or deployment"
            )

        return (
            f"{endpoint}/openai/deployments/{deployment}/embeddings"
            f"?api-version={api_version}"
        )

    def _build_request(self, payload: dict) -> Any:
        # Azure 特化：使用 api-key 头，不使用 Bearer。
        api_key = self._get_api_key()
        if not api_key:
            raise OpenAIEmbeddingError("provider=azure error_type=ConfigError detail=missing api_key")

        endpoint = self._build_embedding_endpoint()
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        return urlrequest.Request(endpoint, data=data, headers=headers, method="POST")

    def _get_model_name(self) -> str:
        # Azure 场景下 model 可以复用 deployment 名称。
        deployment = _read_embedding_option(self.settings, "deployment", default="")
        return str(deployment or "azure-embedding-deployment")
