"""OpenAI-compatible Embedding 实现基类与 OpenAI provider。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """OpenAI Embedding 统一异常。"""


class OpenAICompatibleEmbedding(BaseEmbedding):
    """OpenAI-compatible Embedding 通用实现。

    设计意图：
    1. 将输入预处理、HTTP 调用、响应解析等横切逻辑集中管理。
    2. Azure 等兼容 provider 只需覆写 endpoint / 鉴权细节。
    3. 对空输入、超长输入给出明确且可配置的行为。
    """

    provider_name = "openai-compatible"

    def embed(self, texts: List[str], trace: Any = None) -> List[List[float]]:
        prepared = _prepare_texts(texts, settings=self.settings, provider=self.provider_name)

        payload = {
            "model": self._get_model_name(),
            "input": prepared,
        }

        req = self._build_request(payload)

        try:
            with urlrequest.urlopen(req, timeout=self._get_timeout_seconds()) as resp:
                body = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise OpenAIEmbeddingError(
                f"provider={self.provider_name} error_type=HTTPError status={exc.code} detail={detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise OpenAIEmbeddingError(
                f"provider={self.provider_name} error_type=URLError detail={exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OpenAIEmbeddingError(
                f"provider={self.provider_name} error_type=TimeoutError detail={exc}"
            ) from exc

        try:
            data = json.loads(body)
            return _extract_embeddings(data, provider=self.provider_name)
        except (ValueError, KeyError, TypeError) as exc:
            raise OpenAIEmbeddingError(
                f"provider={self.provider_name} error_type=ResponseParseError detail={exc}"
            ) from exc

    def _build_request(self, payload: Dict[str, Any]) -> urlrequest.Request:
        api_key = self._get_api_key()
        if not api_key:
            raise OpenAIEmbeddingError(
                f"provider={self.provider_name} error_type=ConfigError detail=missing api_key"
            )

        endpoint = self._build_embedding_endpoint()
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        return urlrequest.Request(endpoint, data=data, headers=headers, method="POST")

    def _build_embedding_endpoint(self) -> str:
        base_url = str(_read_embedding_option(self.settings, "base_url", default="https://api.openai.com/v1"))
        return f"{base_url.rstrip('/')}/embeddings"

    def _get_model_name(self) -> str:
        model = _read_embedding_option(self.settings, "model", default="text-embedding-3-small")
        return str(model)

    def _get_api_key(self) -> str:
        api_key = _read_embedding_option(self.settings, "api_key", default="")
        return str(api_key)

    def _get_timeout_seconds(self) -> float:
        timeout = _read_embedding_option(self.settings, "timeout_seconds", default=30)
        return float(timeout)


class OpenAIEmbedding(OpenAICompatibleEmbedding):
    """OpenAI 官方 Embedding provider。"""

    provider_name = "openai"


def _prepare_texts(texts: Sequence[Any], settings: Any, provider: str) -> List[str]:
    """输入预处理：校验类型、空输入和超长输入策略。"""
    if not isinstance(texts, list) or not texts:
        raise OpenAIEmbeddingError(
            f"provider={provider} error_type=ValidationError detail=texts must be non-empty list[str]"
        )

    max_input_chars = int(_read_embedding_option(settings, "max_input_chars", default=8192))
    truncate_input = bool(_read_embedding_option(settings, "truncate_input", default=False))

    prepared: List[str] = []
    for item in texts:
        if is_dataclass(item):
            item = asdict(item)
        if not isinstance(item, str):
            raise OpenAIEmbeddingError(
                f"provider={provider} error_type=ValidationError detail=text item must be str"
            )

        text = item.strip()
        if not text:
            raise OpenAIEmbeddingError(
                f"provider={provider} error_type=ValidationError detail=empty text is not allowed"
            )

        if len(text) > max_input_chars:
            if truncate_input:
                text = text[:max_input_chars]
            else:
                raise OpenAIEmbeddingError(
                    f"provider={provider} error_type=ValidationError detail=text too long"
                )

        prepared.append(text)

    return prepared


def _extract_embeddings(data: Dict[str, Any], provider: str) -> List[List[float]]:
    """解析 OpenAI-compatible Embedding 响应。"""
    items = data.get("data")
    if not isinstance(items, list) or not items:
        raise KeyError(f"provider={provider} missing data")

    # 按 index 排序，确保返回顺序与输入一致。
    sorted_items = sorted(items, key=lambda x: x.get("index", 0))
    vectors: List[List[float]] = []
    for item in sorted_items:
        emb = item.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise KeyError(f"provider={provider} missing embedding")
        vector = []
        for value in emb:
            if not isinstance(value, (int, float)):
                raise TypeError("embedding vector contains non-numeric value")
            vector.append(float(value))
        vectors.append(vector)
    return vectors


def _read_embedding_option(settings: Any, key: str, default: Any) -> Any:
    """从 settings.embedding 读取字段，兼容 dataclass 和 dict。"""
    if hasattr(settings, "embedding") and hasattr(settings.embedding, key):
        value = getattr(settings.embedding, key)
        return default if value is None else value
    if isinstance(settings, dict):
        embedding = settings.get("embedding")
        if isinstance(embedding, dict):
            value = embedding.get(key, default)
            return default if value is None else value
    return default
